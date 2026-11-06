"""A genuine patch-based byte LM, with no subword inference fallback."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from bytepft.config import BytePFTConfig
from bytepft.models.boundaries import BoundaryPredictor, validate_attention_mask
from bytepft.models.decoder import LocalByteDecoder
from bytepft.models.depooling import PatchDepool
from bytepft.models.embeddings import ByteEmbedding
from bytepft.models.encoder import LocalByteEncoder
from bytepft.models.heads import BytePredictionHead
from bytepft.models.pooling import PatchMetadata, PatchPool
from bytepft.tokenization.byte_tokenizer import ByteTokenizer


@dataclass
class ByteModelOutput:
    logits: torch.Tensor
    boundary_logits: torch.Tensor
    boundaries: torch.Tensor
    patch_embeddings: torch.Tensor
    patch_metadata: PatchMetadata
    global_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    loss: torch.Tensor | None = None


class ByteModel(nn.Module):
    def __init__(self, global_adapter: nn.Module, config: BytePFTConfig | None = None) -> None:
        super().__init__()
        self.config = config or BytePFTConfig()
        self.global_adapter = global_adapter
        self.tokenizer = ByteTokenizer()
        hidden = self.config.hidden_size
        global_hidden = global_adapter.hidden_size
        self.embeddings = ByteEmbedding(hidden)
        self.encoder = LocalByteEncoder(hidden, self.config.encoder_layers)
        self.boundary = BoundaryPredictor(
            hidden,
            mode=self.config.boundary_mode,
            patch_size=self.config.patch_size,
            min_patch_size=self.config.min_patch_size,
            max_patch_size=self.config.max_patch_size,
            threshold=self.config.boundary_threshold,
        )
        self.pool = PatchPool(self.config.pooling)
        self.to_global = nn.Linear(hidden, global_hidden, bias=False)
        self.depool = PatchDepool()
        self.decoder = LocalByteDecoder(hidden, global_hidden, self.config.decoder_layers)
        self.head = BytePredictionHead(hidden)
        self.training_status: dict[str, Any] = {
            "stage1_steps": 0,
            "stage2_steps": 0,
            "stage1": "not_started",
            "stage2": "not_started",
            "trained": False,
            "quality_validated": False,
        }
        self.metrics: dict[str, Any] = {}
        self.checkpoint_metadata: dict[str, Any] = {}
        self.patch_statistics: dict[str, int] = {"bytes": 0, "patches": 0}
        # Construction does not change any pretrained parameter values.
        parameter = next(global_adapter.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        oracle_boundaries: torch.Tensor | None = None,
        teacher_patch_states: torch.Tensor | None = None,
    ) -> ByteModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long")
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id)
            if attention_mask is None
            else validate_attention_mask(attention_mask)
        )
        if attention_mask.shape != input_ids.shape or not attention_mask.any(dim=1).all():
            raise ValueError("each sequence must contain valid input, normally beginning with BOS")
        attention_mask = validate_attention_mask(attention_mask)
        embedded = self.embeddings(input_ids, attention_mask)
        encoded = self.encoder(embedded, attention_mask)
        boundary_logits, boundaries = self.boundary(
            encoded,
            attention_mask,
            input_ids=input_ids,
            oracle_boundaries=oracle_boundaries,
        )
        if oracle_boundaries is not None:
            # Stage 1 teacher forcing is an explicit override for alignment experiments.
            boundaries = oracle_boundaries.bool() & attention_mask
        pooled, metadata = self.pool(encoded, boundaries, attention_mask)
        projected = self.to_global(pooled)
        if teacher_patch_states is None:
            contextualized = self.global_adapter(
                projected,
                attention_mask=metadata.patch_mask,
                use_cache=False,
            ).hidden_states
        else:
            if teacher_patch_states.shape != projected.shape:
                raise ValueError("teacher states must align exactly with the oracle patch axis")
            contextualized = teacher_patch_states.detach()
        context = self.depool(contextualized, metadata)
        decoded = self.decoder(encoded, context, attention_mask)
        logits = self.head(decoded)
        loss = None
        if labels is not None:
            from bytepft.training.losses import ByteCrossEntropyLoss

            loss = ByteCrossEntropyLoss()(logits, labels)
        return ByteModelOutput(
            logits,
            boundary_logits,
            boundaries,
            projected,
            metadata,
            contextualized,
            encoded,
            loss,
        )

    def generate(self, prompt: str = "", **kwargs: Any) -> Any:
        from bytepft.inference.generation import generate

        return generate(self, prompt, **kwargs)

    def save_pretrained(self, directory: str) -> None:
        from bytepft.serialization import save_pretrained

        save_pretrained(self, directory)

    def stats(self) -> dict[str, Any]:
        original = self.global_adapter.original_parameter_count
        total = sum(p.numel() for p in self.parameters())
        count = self.patch_statistics["patches"]
        return {
            "parameters": total,
            "original_parameters": original,
            "new_parameters": total - original,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "input_interface": "UTF-8 bytes 0..255; PAD=256, BOS=257, EOS=258",
            "output_interface": "259-way reference head; generation suppresses PAD and BOS",
            "source_model": self.global_adapter.source_model_id,
            "architecture": self.config.architecture,
            "average_bytes_per_patch": self.patch_statistics["bytes"] / count if count else None,
            "patch_count": count or None,
            "training": dict(self.training_status),
            "checkpoint": dict(self.checkpoint_metadata),
            "metrics": dict(self.metrics),
        }
