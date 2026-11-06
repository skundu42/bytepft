"""Explicit two-stage adaptation with deterministic resumable CPU-scale training."""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from bytepft.config import BytePFTTrainingConfig
from bytepft.exceptions import CompatibilityError
from bytepft.models.boundaries import fixed_boundaries
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained, save_pretrained, write_json
from bytepft.training.datasets import (
    ByteDataset,
    collate_documents,
    deterministic_split,
    load_texts,
    source_boundary_targets,
)
from bytepft.training.distillation import collect_patch_log_probs, extract_source_alignment
from bytepft.training.losses import (
    BoundaryLoss,
    ByteCrossEntropyLoss,
    DecoderDistillationLoss,
    EmbeddingAlignmentLoss,
)


class BytePFTTrainer:
    def __init__(
        self,
        model: ByteModel,
        config: BytePFTTrainingConfig,
        *,
        source_tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.config = config
        self.source_tokenizer = source_tokenizer
        self.history: list[dict[str, Any]] = []
        self._random = random.Random(config.seed)
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._active_stage = ""
        self._fingerprint = ""
        self._resume_state: dict[str, Any] | None = None
        self._best_validation = float("inf")
        self._bad_validations = 0
        self._documents_seen = 0
        self._source_tokens_seen = 0
        self._amp_dtype = {
            "fp32": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[model.config.precision]
        if model.config.precision == "fp16" and model.device.type != "cuda":
            raise CompatibilityError("fp16 adaptation requires CUDA AMP; use fp32 or CPU bf16")
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=model.device.type == "cuda" and model.config.precision == "fp16",
        )

    def _aligned_forward(self, batch: dict[str, Any]) -> tuple[Any, dict[str, torch.Tensor]]:
        if self.source_tokenizer is None:
            raise ValueError("Aligned Stage 1 requires the original source tokenizer")
        alignments = [
            extract_source_alignment(text, self.source_tokenizer) for text in batch["texts"]
        ]
        ends = [alignment.token_ends for alignment in alignments]
        targets, boundary_mask = source_boundary_targets(batch["input_ids"], ends)
        oracle = targets.bool() | (batch["input_ids"] == 257) | (batch["input_ids"] == 258)
        adapter = self.model.global_adapter
        hf_config = adapter.model.config
        bos = hf_config.bos_token_id
        eos = hf_config.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        if bos is None or eos is None:
            raise ValueError("Aligned Stage 1 requires explicit source BOS and EOS IDs")
        width = max(len(a.token_ids) for a in alignments) + 2
        teacher_ids = torch.full(
            (len(alignments), width), eos, device=self.model.device, dtype=torch.long
        )
        teacher_mask = torch.zeros_like(teacher_ids, dtype=torch.bool)
        teacher_event_probs = torch.zeros((len(alignments), width - 2), device=self.model.device)
        with torch.no_grad():
            for row, alignment in enumerate(alignments):
                ids = [bos, *alignment.token_ids, eos]
                teacher_ids[row, : len(ids)] = torch.tensor(ids, device=self.model.device)
                teacher_mask[row, : len(ids)] = True
            teacher = adapter.teacher_forward(teacher_ids, teacher_mask)
            teacher_embeddings = adapter.get_input_embeddings()(teacher_ids)
            for row, alignment in enumerate(alignments):
                count = len(alignment.token_ids)
                if count:
                    token_ids = teacher_ids[row, 1 : count + 1]
                    teacher_event_probs[row, :count] = (
                        teacher.logits[row, :count]
                        .float()
                        .log_softmax(-1)
                        .gather(-1, token_ids[:, None])
                        .squeeze(-1)
                    )
        output = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            oracle_boundaries=oracle,
            teacher_patch_states=teacher.hidden_states,
        )
        event_probs, event_mask = collect_patch_log_probs(
            output.logits,
            output.boundary_logits,
            batch["input_ids"],
            ends,
        )
        losses = {
            "boundary": BoundaryLoss()(output.boundary_logits, targets, boundary_mask),
            "embedding": EmbeddingAlignmentLoss()(
                output.patch_embeddings, teacher_embeddings, teacher_mask
            ),
            "distillation": DecoderDistillationLoss(self.config.stage1.temperature)(
                event_probs, teacher_event_probs, event_mask
            ),
            "byte_ce": ByteCrossEntropyLoss()(output.logits, batch["labels"]),
        }
        self._source_tokens_seen += sum(len(a.token_ids) for a in alignments)
        return output, losses

    def _loss(
        self, batch: dict[str, Any], stage: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
        batch = {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if stage == "stage1" and self.config.stage1.mode == "aligned":
            output, losses = self._aligned_forward(batch)
        else:
            output = self.model(batch["input_ids"], batch["attention_mask"])
            targets = fixed_boundaries(
                batch["attention_mask"], self.model.config.patch_size, batch["input_ids"]
            )
            losses = {
                "byte_ce": ByteCrossEntropyLoss()(output.logits, batch["labels"]),
                "boundary": BoundaryLoss()(
                    output.boundary_logits,
                    targets.float(),
                    batch["attention_mask"] & (batch["input_ids"] < 256),
                ),
            }
        if stage == "stage1":
            weights = self.config.stage1
            total = sum(
                value * getattr(weights, f"{name}_weight") for name, value in losses.items()
            )
        else:
            # Compression regularization is a baseline fixed-target surrogate, not paper Stage 2.
            total = losses["byte_ce"] + losses["boundary"]
        return total, losses, output

    def _start_stage(self, stage: str, steps: int) -> None:
        self._active_stage = stage
        if stage == "stage1" or not self.config.stage2.unfreeze_global:
            self.model.global_adapter.freeze()
        else:
            self.model.global_adapter.unfreeze()
        global_ids = {id(p) for p in self.model.global_adapter.parameters()}
        local = [p for p in self.model.parameters() if p.requires_grad and id(p) not in global_ids]
        groups: list[dict[str, Any]] = [{"params": local, "lr": self.config.learning_rate}]
        global_parameters = [p for p in self.model.global_adapter.parameters() if p.requires_grad]
        if global_parameters:
            groups.append(
                {
                    "params": global_parameters,
                    "lr": self.config.stage2.global_learning_rate or self.config.learning_rate,
                }
            )
        self._optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

        def scale(step: int) -> float:
            if step < self.config.warmup_steps:
                return (step + 1) / max(1, self.config.warmup_steps)
            return max(0.0, (steps - step) / max(1, steps - self.config.warmup_steps))

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, scale)
        if self._resume_state and self._resume_state["active_stage"] == stage:
            self._optimizer.load_state_dict(self._resume_state["optimizer"])
            self._scheduler.load_state_dict(self._resume_state["scheduler"])
            self._scaler.load_state_dict(self._resume_state.get("scaler", {}))
            self._resume_state = None
        self.model.train()

    def save_checkpoint(self, directory: str | Path) -> None:
        if self._optimizer is None or self._scheduler is None:
            raise RuntimeError("No training stage is active")
        path = Path(directory)
        save_pretrained(self.model, path)
        write_json(path / "training_config.json", self.config.to_dict())
        state = {
            "active_stage": self._active_stage,
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "batch_rng": self._random.getstate(),
            "fingerprint": self._fingerprint,
            "history": self.history,
            "best_validation": self._best_validation,
            "bad_validations": self._bad_validations,
            "documents_seen": self._documents_seen,
            "source_tokens_seen": self._source_tokens_seen,
            "scaler": self._scaler.state_dict(),
        }
        if self.model.device.type == "cuda":
            state["device_rng"] = torch.cuda.get_rng_state(self.model.device)
        elif self.model.device.type == "mps":
            state["device_rng"] = torch.mps.get_rng_state()
        temporary = path / "trainer_state.pt.tmp"
        torch.save(state, temporary)
        os.replace(temporary, path / "trainer_state.pt")

    def _resume(self, directory: str | Path) -> None:
        path = Path(directory)
        previous = json.loads((path / "training_config.json").read_text())
        # Step targets, optimizer settings and schedule must match for exact resumption.
        current = self.config.to_dict()
        previous.pop("output_dir", None)
        current.pop("output_dir", None)
        if previous != current:
            raise ValueError(
                "Resume requires the original training configuration (except output_dir)"
            )
        restored = load_pretrained(path, device=str(self.model.device))
        if (
            restored.config != self.model.config
            or restored.global_adapter.hf_config != self.model.global_adapter.hf_config
        ):
            raise ValueError("Checkpoint model architecture differs from this trainer")
        self.model.to(dtype=next(restored.parameters()).dtype)
        self.model.load_state_dict(restored.state_dict())
        self.model.training_status = restored.training_status
        self.model.metrics = restored.metrics
        self.model.patch_statistics = restored.patch_statistics
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=True)
        if state["fingerprint"] != self._fingerprint:
            raise ValueError("Training/validation documents differ from the resumed checkpoint")
        torch.set_rng_state(state["torch_rng"])
        if "device_rng" in state:
            if self.model.device.type == "cuda":
                torch.cuda.set_rng_state(state["device_rng"], self.model.device)
            elif self.model.device.type == "mps":
                torch.mps.set_rng_state(state["device_rng"])
        self._random.setstate(state["batch_rng"])
        self.history = state["history"]
        self._best_validation = state["best_validation"]
        self._bad_validations = state["bad_validations"]
        self._documents_seen = state["documents_seen"]
        self._source_tokens_seen = state["source_tokens_seen"]
        self._resume_state = state

    def fit(
        self,
        dataset: Any,
        *,
        validation_dataset: Any = None,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        from bytepft.evaluation.metrics import evaluate_model

        if self.config.stage1.steps + self.config.stage2.steps < 1:
            raise ValueError("Request at least one Stage 1 or Stage 2 optimizer step")
        if self.model.config.boundary_mode == "oracle":
            raise ValueError(
                "Oracle mode is for forward diagnostics; configure fixed or learned "
                "for a trainable inference workflow"
            )
        documents = load_texts(dataset, text_field=self.config.text_field)
        if not documents:
            raise ValueError("Training dataset contains no documents")
        if validation_dataset is None:
            train_texts, validation_texts = deterministic_split(
                documents, self.config.validation_fraction, seed=self.config.seed
            )
        else:
            train_texts, validation_texts = (
                documents,
                load_texts(validation_dataset, text_field=self.config.text_field),
            )
        self._fingerprint = hashlib.sha256(
            json.dumps([train_texts, validation_texts], ensure_ascii=False).encode()
        ).hexdigest()
        train = ByteDataset(train_texts, max_bytes=self.config.max_bytes)
        if not any(text.encode("utf-8") for text in train.texts):
            raise ValueError(
                "Training data contains no raw bytes after truncation; increase max_bytes"
            )
        if validation_texts and not any(
            ByteDataset(validation_texts, max_bytes=self.config.max_bytes).texts
        ):
            raise ValueError(
                "Validation data contains no raw bytes after truncation; increase max_bytes"
            )
        # AMP keeps master parameters and Adam moments in fp32. This costs more
        # memory than loading half-precision weights alone, but avoids half-state updates.
        if self._amp_dtype is not None:
            self.model.float()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None:
            self._resume(resume_from)
        else:
            torch.manual_seed(self.config.seed)
        if all(
            self.model.training_status[f"{stage}_steps"] >= target
            for stage, target in (
                ("stage1", self.config.stage1.steps),
                ("stage2", self.config.stage2.steps),
            )
        ):
            self.model.eval()
            return self.history
        if self.config.activation_checkpointing:
            self.model.global_adapter.model.gradient_checkpointing_enable()
        for stage, target in (
            ("stage1", self.config.stage1.steps),
            ("stage2", self.config.stage2.steps),
        ):
            completed = self.model.training_status[f"{stage}_steps"]
            if target <= completed:
                continue
            self._start_stage(stage, target)
            self.model.training_status[stage] = "running"
            self._best_validation = float("inf") if completed == 0 else self._best_validation
            self._bad_validations = 0 if completed == 0 else self._bad_validations
            assert self._optimizer is not None and self._scheduler is not None
            for step in range(completed, target):
                self._optimizer.zero_grad(set_to_none=True)
                observed: dict[str, float] = {}
                for _ in range(self.config.gradient_accumulation_steps):
                    batch = collate_documents(
                        [
                            train[self._random.randrange(len(train))]
                            for _ in range(self.config.batch_size)
                        ]
                    )
                    with torch.autocast(
                        device_type=self.model.device.type,
                        dtype=self._amp_dtype or torch.float32,
                        enabled=self._amp_dtype is not None,
                    ):
                        total, losses, result = self._loss(batch, stage)
                    if not torch.isfinite(total):
                        raise FloatingPointError(f"Nonfinite {stage} loss at step {step + 1}")
                    self._scaler.scale(total / self.config.gradient_accumulation_steps).backward()
                    for name, value in {"loss": total, **losses}.items():
                        observed[name] = (
                            observed.get(name, 0.0)
                            + float(value.detach()) / self.config.gradient_accumulation_steps
                        )
                    self.model.patch_statistics["bytes"] += sum(batch["byte_counts"])
                    for row in range(batch["input_ids"].shape[0]):
                        ordinary = (batch["input_ids"][row] < 256).to(
                            result.patch_metadata.byte_to_patch.device
                        )
                        self.model.patch_statistics["patches"] += int(
                            result.patch_metadata.byte_to_patch[row][ordinary].unique().numel()
                        )
                    self._documents_seen += len(batch["texts"])
                parameters = [
                    p for p in self.model.parameters() if p.requires_grad and p.grad is not None
                ]
                self._scaler.unscale_(self._optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.max_grad_norm, error_if_nonfinite=True
                )
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._scheduler.step()
                self.model.training_status[f"{stage}_steps"] = step + 1
                record = {"stage": stage, "step": step + 1, **observed, "grad_norm": float(norm)}
                self.history.append(record)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                if validation_texts and (step + 1) % self.config.validation_interval == 0:
                    metrics = evaluate_model(
                        self.model,
                        validation_texts,
                        max_bytes=self.config.max_bytes,
                        batch_size=self.config.batch_size,
                    )
                    self.model.metrics["validation"] = metrics
                    if metrics["nll"] < self._best_validation:
                        self._best_validation = metrics["nll"]
                        self._bad_validations = 0
                    else:
                        self._bad_validations += 1
                if (step + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(output_dir / f"{stage}-step-{step + 1}")
                patience = self.config.early_stopping_patience
                if patience and self._bad_validations >= patience:
                    self.model.training_status[stage] = "early_stopped"
                    break
            else:
                self.model.training_status[stage] = "completed"
            self.model.training_status["trained"] = True
            self.save_checkpoint(output_dir / f"{stage}-final")
        self.model.eval()
        evaluation = validation_texts or train_texts
        self.model.metrics["validation" if validation_texts else "train"] = evaluate_model(
            self.model,
            evaluation,
            max_bytes=self.config.max_bytes,
            batch_size=self.config.batch_size,
        )
        self.model.metrics["training_counts"] = {
            "documents_seen": self._documents_seen,
            "source_tokens_seen": self._source_tokens_seen
            if self.config.stage1.mode == "aligned"
            else None,
            "data_sha256": self._fingerprint,
            "seed": self.config.seed,
        }
        self.save_checkpoint(output_dir / "checkpoint-final")
        save_pretrained(self.model, output_dir)
        return self.history
