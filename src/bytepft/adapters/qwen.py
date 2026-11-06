"""Qwen3 integration using the Hugging Face transformer implementation.

This module neither rewrites Qwen blocks nor changes pretrained parameters at
construction time. Source token embeddings are retained for Stage 1 alignment;
byte inference enters the backbone through ``inputs_embeds``.
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from bytepft.adapters.base import GlobalAdapter, GlobalOutput, TeacherOutput
from bytepft.exceptions import CompatibilityError


class QwenGlobalAdapter(GlobalAdapter):
    """Validated wrapper around a dense Hugging Face Qwen3 causal LM.

    ``attention_mask`` includes cached positions when ``past_key_values`` is
    provided. Padded cached batches must supply that complete mask on every
    step; no cache contains an implicit history of padding masks.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        source_model_id: str | None = None,
        source_revision: str | None = None,
        kind: str = "qwen3",
    ) -> None:
        super().__init__()
        if getattr(model.config, "model_type", None) != "qwen3":
            raise CompatibilityError("Only dense Qwen3 causal language models are supported.")
        if not isinstance(model, Qwen3ForCausalLM):
            raise CompatibilityError("Expected the Hugging Face Qwen3ForCausalLM implementation.")
        backbone = model.get_decoder()
        if not all(hasattr(backbone, name) for name in ("layers", "norm", "rotary_emb")):
            raise CompatibilityError(
                "This Qwen3 backbone does not expose the expected decoder interface."
            )
        embedding = model.get_input_embeddings()
        projection = model.get_output_embeddings()
        if not isinstance(embedding, nn.Embedding) or not isinstance(projection, nn.Linear):
            raise CompatibilityError(
                "Qwen3 input embeddings and output projection are incompatible."
            )
        hidden_size = model.config.hidden_size
        if embedding.embedding_dim != hidden_size or projection.in_features != hidden_size:
            raise CompatibilityError(
                "Qwen3 embedding/projection dimensions disagree with hidden_size."
            )
        if model.config.head_dim % 2:
            raise CompatibilityError("Qwen3 rotary attention requires an even head_dim.")
        if kind not in {"qwen3", "tiny"}:
            raise ValueError("Adapter kind must be 'qwen3' or 'tiny'.")
        self.model = model
        self.hidden_size = hidden_size
        self.source_model_id = source_model_id
        self.source_revision = source_revision
        self.kind = kind
        self.original_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self._frozen = False

    @property
    def backbone(self) -> nn.Module:
        return self.model.get_decoder()

    @property
    def hf_config(self) -> dict[str, Any]:
        return self.model.config.to_dict()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.get_output_embeddings()

    def freeze(self) -> None:
        self._frozen = True
        super().freeze()

    def unfreeze(self) -> None:
        self._frozen = False
        super().unfreeze()

    def train(self, mode: bool = True) -> "QwenGlobalAdapter":
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @classmethod
    def from_config(
        cls,
        config: Qwen3Config | dict[str, Any],
        *,
        source_model_id: str | None = None,
        source_revision: str | None = None,
        kind: str = "qwen3",
    ) -> "QwenGlobalAdapter":
        """Construct random weights, used for offline tests and checkpoint loading."""
        if isinstance(config, dict):
            if config.get("model_type", "qwen3") != "qwen3":
                raise CompatibilityError("Only dense Qwen3 configurations are supported.")
            config = Qwen3Config.from_dict(config)
        if not isinstance(config, Qwen3Config):
            raise CompatibilityError(
                "Expected Qwen3Config; other model adapters are not implemented."
            )
        config._attn_implementation = "sdpa"
        return cls(
            Qwen3ForCausalLM(config),
            source_model_id=source_model_id,
            source_revision=source_revision,
            kind=kind,
        )

    @classmethod
    def from_pretrained(
        cls,
        identity: str,
        *,
        device: str | torch.device = "cpu",
        precision: str = "fp32",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "QwenGlobalAdapter":
        """Load a supported checkpoint without executing repository-provided code."""
        dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        if precision not in dtypes:
            raise ValueError("precision must be fp32, bf16, or fp16.")
        device = torch.device(device)
        if device.type == "cpu" and precision == "fp16":
            raise ValueError("fp16 training/inference on CPU is unsupported; use fp32 or bf16.")
        kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": local_files_only}
        if revision is not None:
            kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(identity, **kwargs)
        if getattr(config, "model_type", None) != "qwen3":
            raise CompatibilityError(
                f"Checkpoint {identity!r} has model_type={config.model_type!r}; "
                "only dense Qwen3 is implemented."
            )
        model = AutoModelForCausalLM.from_pretrained(
            identity,
            config=config,
            torch_dtype=dtypes[precision],
            attn_implementation="sdpa",
            use_safetensors=True,
            **kwargs,
        ).to(device)
        return cls(
            model,
            source_model_id=identity,
            source_revision=getattr(config, "_commit_hash", None) or revision,
        )

    def _positions(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length = inputs.shape[:2]
        past_length = 0 if past_key_values is None else int(past_key_values.get_seq_length())
        total_length = past_length + query_length
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, total_length, dtype=torch.long, device=inputs.device
            )
        if attention_mask.shape != (batch_size, total_length):
            raise ValueError(
                "attention_mask must have shape [batch, cached_patches + new_patches], "
                f"expected {(batch_size, total_length)}, got {tuple(attention_mask.shape)}."
            )
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only zeros and ones.")
        attention_mask = attention_mask.to(device=inputs.device, dtype=torch.long)
        if position_ids is None:
            position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)[:, -query_length:]
        elif position_ids.shape != (batch_size, query_length):
            raise ValueError("position_ids must have shape [batch, new_patches].")
        position_ids = position_ids.to(device=inputs.device, dtype=torch.long)
        valid_positions = position_ids[attention_mask[:, -query_length:].bool()]
        if valid_positions.numel() and (
            int(valid_positions.min()) < 0
            or int(valid_positions.max()) >= self.model.config.max_position_embeddings
        ):
            raise ValueError("Patch positions exceed the Qwen3 context window.")
        return attention_mask, position_ids

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> GlobalOutput:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError(f"inputs_embeds must have shape [batch, patches, {self.hidden_size}].")
        if not inputs_embeds.is_floating_point():
            raise ValueError("inputs_embeds must be floating point.")
        if inputs_embeds.shape[1] == 0:
            return GlobalOutput(inputs_embeds, past_key_values)
        attention_mask, position_ids = self._positions(
            inputs_embeds, attention_mask, past_key_values, position_ids
        )
        output = self.backbone(
            inputs_embeds=inputs_embeds.to(dtype=self.get_input_embeddings().weight.dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = output.last_hidden_state
        if hidden_states.shape != inputs_embeds.shape:
            raise CompatibilityError("Qwen3 returned an unexpected patch-state shape.")
        hidden_states = hidden_states * attention_mask[:, -inputs_embeds.shape[1] :].unsqueeze(-1)
        return GlobalOutput(hidden_states, output.past_key_values)

    @torch.no_grad()
    def teacher_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> TeacherOutput:
        """Evaluate source-token probabilities without retaining teacher gradients."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Teacher input_ids must have nonempty shape [batch, source_tokens].")
        attention_mask, position_ids = self._positions(input_ids, attention_mask, None, None)
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = output.last_hidden_state
            logits = self.get_output_embeddings()(hidden_states)
        finally:
            self.model.train(was_training)
        return TeacherOutput(logits, hidden_states)


class TinyGlobalAdapter(QwenGlobalAdapter):
    """Randomly initialized Qwen3 transformer for small, CPU-only experiments."""

    def __init__(
        self,
        model: Qwen3ForCausalLM | None = None,
        *,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        intermediate_size: int = 64,
        vocab_size: int = 64,
        source_model_id: str | None = None,
        source_revision: str | None = None,
        kind: str = "tiny",
    ) -> None:
        if model is None:
            if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
                raise ValueError(
                    "Tiny hidden_size must divide into even-dimensional attention heads."
                )
            config = Qwen3Config(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=hidden_size // num_attention_heads,
                vocab_size=vocab_size,
                attention_dropout=0.0,
                max_position_embeddings=4096,
                tie_word_embeddings=True,
            )
            config._attn_implementation = "sdpa"
            model = Qwen3ForCausalLM(config)
        super().__init__(
            model, source_model_id=source_model_id, source_revision=source_revision, kind=kind
        )
