"""A byte vocabulary with separate padding and document control symbols."""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


class IncrementalUTF8Decoder:
    """Decode complete UTF-8 characters while buffering an unfinished suffix.

    ``feed`` may return an empty string for the first bytes of a multibyte
    character. ``finish`` applies the chosen error policy to an incomplete
    final character; the default is to raise, rather than silently lose bytes.
    """

    def __init__(self, errors: str = "strict") -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)

    def feed(self, values: bytes | bytearray | memoryview | Iterable[int]) -> str:
        return self._decoder.decode(bytes(values), final=False)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)

    @property
    def pending_bytes(self) -> bytes:
        return self._decoder.getstate()[0]

    def reset(self) -> None:
        self._decoder.reset()


class ByteTokenizer:
    """Map ordinary bytes to IDs 0..255 and controls to IDs 256..258.

    This class implements the commonly used callable Hugging Face tokenizer
    surface (batch IDs, masks and optional PyTorch tensors). It deliberately
    does not pretend to implement the full ``PreTrainedTokenizer`` protocol.
    """

    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    all_special_ids = [256, 257, 258]

    def __len__(self) -> int:
        return self.vocab_size

    def encode_bytes(
        self,
        value: bytes | bytearray | memoryview,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("encode_bytes expects bytes, bytearray or memoryview")
        return (
            ([self.bos_token_id] if add_bos else [])
            + list(bytes(value))
            + ([self.eos_token_id] if add_eos else [])
        )

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode expects a Unicode string; use encode_bytes for raw bytes")
        return self.encode_bytes(
            text.encode("utf-8"),
            add_bos=add_bos or add_special_tokens,
            add_eos=add_eos or add_special_tokens,
        )

    def decode_bytes(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> bytes:
        ordinary: list[int] = []
        for raw_value in ids:
            # torch scalar IDs expose __index__, as do Python and NumPy integers.
            try:
                value = raw_value.__index__()
            except (AttributeError, TypeError) as error:
                raise TypeError("Token IDs must be integers") from error
            if 0 <= value <= 255:
                ordinary.append(value)
            elif value in self.all_special_ids:
                if not skip_special_tokens:
                    raise ValueError("Control IDs cannot be decoded as raw bytes")
            else:
                raise ValueError(f"Unknown byte token ID: {value}")
        return bytes(ordinary)

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        return self.decode_bytes(ids, skip_special_tokens=skip_special_tokens).decode(
            "utf-8", errors=errors
        )

    def batch_decode(self, ids: Iterable[Iterable[int]], **kwargs: Any) -> list[str]:
        return [self.decode(row, **kwargs) for row in ids]

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        padding: bool | str = True,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> dict[str, Any]:
        if padding not in (True, False, "longest", "max_length"):
            raise ValueError("padding must be False, True, 'longest', or 'max_length'")
        if return_tensors not in (None, "pt"):
            raise ValueError("Only return_tensors='pt' is supported")
        if max_length is not None and (
            isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 0
        ):
            raise ValueError("max_length must be a nonnegative integer")
        if padding == "max_length" and max_length is None:
            raise ValueError("padding='max_length' requires max_length")
        scalar = isinstance(text, str)
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(item, add_special_tokens=add_special_tokens) for item in texts]
        if truncation:
            if max_length is None:
                raise ValueError("truncation=True requires max_length")
            # Tokenization is lossless by default. Explicit truncation can split
            # a Unicode character, so decode using the incremental decoder.
            rows = [row[:max_length] for row in rows]
        target = max((len(row) for row in rows), default=0)
        if padding == "max_length":
            target = max_length  # type: ignore[assignment]
            if any(len(row) > target for row in rows):
                raise ValueError("Input exceeds max_length; enable truncation explicitly")
        if padding:
            masks = [[1] * len(row) + [0] * (target - len(row)) for row in rows]
            rows = [row + [self.pad_token_id] * (target - len(row)) for row in rows]
        else:
            masks = [[1] * len(row) for row in rows]
        if return_tensors == "pt":
            import torch

            if len({len(row) for row in rows}) > 1:
                raise ValueError("Tensor batches need equal lengths; set padding=True")
            shape = (len(rows), target if padding else (len(rows[0]) if rows else 0))
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long).reshape(shape),
                "attention_mask": torch.tensor(masks, dtype=torch.long).reshape(shape),
            }
        return {
            "input_ids": rows[0] if scalar else rows,
            "attention_mask": masks[0] if scalar else masks,
        }

    def save_pretrained(self, directory: str | Path) -> tuple[str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "byte_tokenizer.json"
        target.write_text(
            json.dumps(
                {
                    "type": "utf8_bytes",
                    "version": 1,
                    "vocab_size": self.vocab_size,
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return (str(target),)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ByteTokenizer:
        config = json.loads((Path(directory) / "byte_tokenizer.json").read_text("utf-8"))
        expected = {
            "type": "utf8_bytes",
            "version": 1,
            "vocab_size": cls.vocab_size,
            "pad_token_id": cls.pad_token_id,
            "bos_token_id": cls.bos_token_id,
            "eos_token_id": cls.eos_token_id,
        }
        if config != expected:
            raise ValueError("Unsupported byte tokenizer configuration or modified byte mapping")
        return cls()
