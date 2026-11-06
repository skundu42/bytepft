import pytest
import torch

from bytepft import BytePFT, BytePFTConfig, BytePFTTrainingConfig, Stage1Config, Stage2Config
from bytepft.evaluation import evaluate_model
from bytepft.training.trainer import BytePFTTrainer


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model():
    return BytePFT.from_tiny(
        config=BytePFTConfig(
            hidden_size=16,
            encoder_layers=1,
            decoder_layers=1,
            bytes_per_patch=2,
        ),
        global_hidden_size=16,
        seed=29,
        device="cpu",
    ).model


def test_activation_checkpointing_runs_real_stage2_backward_and_updates_global(tmp_path):
    model = make_model()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.global_adapter.named_parameters()
    }
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=0),
        stage2=Stage2Config(steps=1, unfreeze_global=True),
        activation_checkpointing=True,
        batch_size=2,
        learning_rate=0.01,
        max_bytes=16,
        validation_fraction=0,
        output_dir=str(tmp_path),
    )
    history = BytePFTTrainer(model, config).fit(["abcabc", "bcabca"])
    assert model.global_adapter.model.is_gradient_checkpointing
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["loss"]))
    assert torch.isfinite(torch.tensor(history[0]["grad_norm"]))
    gradients = [
        parameter.grad
        for parameter in model.global_adapter.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.global_adapter.named_parameters()
    )


@pytest.mark.parametrize("checkpoint", ["stage1-step-2", "stage1-final", "checkpoint-final"])
def test_real_validation_plateau_triggers_early_stopping_at_second_step(tmp_path, checkpoint):
    model = make_model()
    documents = ["abcabc"] * 2
    initial = evaluate_model(model, documents, max_bytes=16, batch_size=2)
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=5, mode="diagnostic"),
        stage2=Stage2Config(steps=0),
        batch_size=2,
        # Below the fp32 update resolution of ordinary nonzero weights. Zero
        # biases may receive tiny values, which cannot change these fp32 logits.
        learning_rate=1e-30,
        max_bytes=16,
        validation_fraction=0,
        validation_interval=1,
        early_stopping_patience=1,
        checkpoint_interval=1,
        output_dir=str(tmp_path),
    )
    history = BytePFTTrainer(model, config).fit(documents, validation_dataset=documents)
    assert [record["step"] for record in history] == [1, 2]
    assert model.training_status["stage1"] == "early_stopped"
    assert model.training_status["stage1_steps"] == 2
    assert model.training_status["stage2"] == "not_started"
    assert model.training_status["trained"]
    assert model.metrics["validation"] == initial
    assert (tmp_path / "stage1-final" / "trainer_state.pt").exists()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    resumed = make_model()
    resumed_history = BytePFTTrainer(resumed, config).fit(
        documents,
        validation_dataset=documents,
        resume_from=tmp_path / checkpoint,
    )
    assert resumed_history == history
    assert resumed.training_status["stage1"] == "early_stopped"
    assert resumed.training_status["stage1_steps"] == 2
    assert resumed.training_status["trained"]
    assert resumed.metrics == model.metrics
    for name, tensor in resumed.state_dict().items():
        assert torch.equal(tensor, before[name])
