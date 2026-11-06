"""Byte-only greedy/sampled generation with reference and incremental paths."""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from bytepft.exceptions import NotAdaptedError
from bytepft.inference.cache import GenerationCache
from bytepft.tokenization.byte_tokenizer import IncrementalUTF8Decoder

if TYPE_CHECKING:
    from bytepft.models.byte_model import ByteModel


@dataclass
class GenerationResult:
    """Observed bytes and timing; no model-quality or speedup claims are implied."""

    text: str
    data: bytes
    generated_ids: list[int]
    generated_bytes: int
    prompt_bytes: int
    prefill_seconds: float
    time_to_first_byte_seconds: float | None
    generation_seconds: float
    elapsed_seconds: float
    bytes_per_second: float
    global_calls: int
    decode_global_calls: int
    global_calls_per_generated_byte: float | None
    stopped_on_eos: bool
    incomplete_utf8_suffix: bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_hex"] = result.pop("data").hex()
        result["incomplete_utf8_suffix_hex"] = result.pop("incomplete_utf8_suffix").hex()
        return result


def _synchronize(model: ByteModel) -> None:
    # Do not initialize or query a GPU when running the CPU reference path.
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elif model.device.type == "mps":
        torch.mps.synchronize()


def create_cache(model: ByteModel) -> GenerationCache:
    parameter = model.embeddings.weight
    return GenerationCache(
        global_context=parameter.new_zeros((1, 1, model.global_adapter.hidden_size))
    )


def incremental_logits(model: ByteModel, input_id: int, cache: GenerationCache) -> torch.Tensor:
    """Observe one byte/control ID and return logits for its successor.

    The caller must disable gradients and put the model in evaluation mode.
    Exposed to make cached/full numerical comparisons independently testable.
    """
    if isinstance(input_id, bool) or not isinstance(input_id, int) or not 0 <= input_id <= 258:
        raise ValueError("input_id must be an integer byte or control ID in [0,258].")
    if input_id == model.tokenizer.pad_token_id:
        raise ValueError("Padding cannot be observed in an individual generation stream.")
    if model.config.boundary_mode == "oracle":
        raise ValueError(
            "Oracle boundaries require a source tokenizer; use fixed or learned inference."
        )
    ids = torch.tensor([[input_id]], dtype=torch.long, device=model.device)
    embedded = model.embeddings(ids)
    encoded, cache.encoder_state = model.encoder.step(embedded, cache.encoder_state)
    cache.patch_length += 1
    cache.patch_sum = encoded if cache.patch_sum is None else cache.patch_sum + encoded
    control = input_id in (model.tokenizer.bos_token_id, model.tokenizer.eos_token_id)
    if model.config.boundary_mode == "fixed":
        complete = control or cache.patch_length >= model.boundary.patch_size
    else:
        score = float(model.boundary.scores(encoded).sigmoid().item())
        complete = (
            control
            or cache.patch_length >= model.boundary.max_patch_size
            or (
                cache.patch_length >= model.boundary.min_patch_size
                and score >= model.boundary.threshold
            )
        )
    if complete:
        pooled = encoded if model.pool.mode == "last" else cache.patch_sum / cache.patch_length
        output = model.global_adapter(
            model.to_global(pooled), past_key_values=cache.global_cache, use_cache=True
        )
        cache.global_context = output.hidden_states
        cache.global_cache = output.past_key_values
        cache.patches_processed += 1
        cache.global_calls += 1
        cache.patch_sum = None
        cache.patch_length = 0
    decoded, cache.decoder_state = model.decoder.step(
        encoded, cache.global_context, cache.decoder_state
    )
    return model.head(decoded)[0, -1]


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    values = logits.detach().float().cpu().clone()
    if values.ndim != 1 or values.numel() != 259 or not torch.isfinite(values).all():
        raise ValueError("Expected 259 finite byte/control logits.")
    values[256:258] = -torch.inf
    if temperature == 0:
        return int(values.argmax().item())
    values = (values - values.max()) / temperature
    if top_k > 0:
        threshold = values.topk(min(top_k, values.numel())).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    if top_p < 1:
        sorted_values, indices = values.sort(descending=True)
        remove = sorted_values.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values[indices[remove]] = -torch.inf
    return int(torch.multinomial(values.softmax(-1), 1, generator=generator).item())


@torch.no_grad()
def generate(
    model: ByteModel,
    prompt: str = "",
    *,
    max_new_bytes: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    use_cache: bool = True,
    return_bytes: bool = False,
    return_details: bool = False,
    seed: int | None = None,
) -> str | bytes | GenerationResult:
    """Generate with the byte head, returning prompt plus completion by default.

    ``temperature=0`` selects greedy decoding. Invalid UTF-8 is replaced in the
    text view; an incomplete final character remains buffered. ``return_bytes``
    and ``return_details`` preserve every sampled raw byte, including that suffix.
    The cached implementation is for one stream and prefills sequentially.
    """
    if not model.training_status.get("trained", False):
        raise NotAdaptedError(
            "Adaptation has not completed; train the byte components before generation."
        )
    if model.config.boundary_mode == "oracle":
        raise ValueError("Oracle patching is training-only; configure fixed or learned boundaries.")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a Unicode string.")
    if isinstance(max_new_bytes, bool) or not isinstance(max_new_bytes, int) or max_new_bytes < 0:
        raise ValueError("max_new_bytes must be a nonnegative integer.")
    if (
        not isinstance(temperature, (float, int))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative (zero means greedy).")
    if not isinstance(top_p, (float, int)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1].")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer; zero disables top-k filtering.")
    if return_bytes and return_details:
        raise ValueError("Choose return_bytes or return_details, not both.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(secrets.randbits(63) if seed is None else seed)
    prompt_data = prompt.encode("utf-8")
    observed = [model.tokenizer.bos_token_id, *prompt_data]
    generated: list[int] = []
    prefill_seconds = 0.0
    first_byte_time = None
    stopped_on_eos = False
    global_calls = 0
    decode_global_calls = 0
    cache = create_cache(model)
    was_training = model.training
    model.eval()
    _synchronize(model)
    started = time.perf_counter()
    decode_started = started
    try:
        if max_new_bytes:
            if use_cache:
                for input_id in observed:
                    logits = incremental_logits(model, input_id, cache)
                global_calls = cache.global_calls
            else:
                inputs = torch.tensor([observed], dtype=torch.long, device=model.device)
                logits = model(inputs).logits[0, -1]
                global_calls = 1
            _synchronize(model)
            decode_started = time.perf_counter()
            prefill_seconds = decode_started - started
            prefill_global_calls = global_calls
            for position in range(max_new_bytes):
                selected = _sample(logits, temperature, top_p, top_k, generator)
                if selected == model.tokenizer.eos_token_id:
                    stopped_on_eos = True
                    break
                generated.append(selected)
                observed.append(selected)
                if first_byte_time is None:
                    first_byte_time = time.perf_counter() - started
                if position + 1 == max_new_bytes:
                    break
                if use_cache:
                    logits = incremental_logits(model, selected, cache)
                    global_calls = cache.global_calls
                else:
                    inputs = torch.tensor([observed], dtype=torch.long, device=model.device)
                    logits = model(inputs).logits[0, -1]
                    global_calls += 1
            decode_global_calls = global_calls - prefill_global_calls
        _synchronize(model)
        finished = time.perf_counter()
    finally:
        model.train(was_training)
    generation_seconds = finished - decode_started
    raw = prompt_data + bytes(generated)
    decoder = IncrementalUTF8Decoder(errors="replace")
    text = decoder.feed(raw)
    result = GenerationResult(
        text=text,
        data=raw,
        generated_ids=generated,
        generated_bytes=len(generated),
        prompt_bytes=len(prompt_data),
        prefill_seconds=prefill_seconds,
        time_to_first_byte_seconds=first_byte_time,
        generation_seconds=generation_seconds,
        elapsed_seconds=finished - started,
        bytes_per_second=len(generated) / generation_seconds if generation_seconds > 0 else 0.0,
        global_calls=global_calls,
        decode_global_calls=decode_global_calls,
        global_calls_per_generated_byte=(
            decode_global_calls / len(generated) if generated else None
        ),
        stopped_on_eos=stopped_on_eos,
        incomplete_utf8_suffix=decoder.pending_bytes,
    )
    if return_details:
        return result
    return result.data if return_bytes else result.text
