import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config

from bytepft.adapters import QwenGlobalAdapter, TinyGlobalAdapter
from bytepft.exceptions import CompatibilityError


def test_forward_shapes_padding_and_causality():
    torch.manual_seed(1)
    adapter = TinyGlobalAdapter().eval()
    inputs = torch.randn(2, 6, adapter.hidden_size)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    result = adapter(inputs, mask).hidden_states
    assert result.shape == inputs.shape
    assert torch.isfinite(result).all()
    assert not result[1, 3:].any()
    changed = inputs.clone()
    changed[:, 3:] += 20
    torch.testing.assert_close(adapter(changed, mask).hidden_states[:, :3], result[:, :3])
    torch.testing.assert_close(adapter(inputs[1:2, :3]).hidden_states, result[1:2, :3])


@pytest.mark.parametrize("with_padding", [False, True])
def test_cache_matches_full_transformer(with_padding):
    torch.manual_seed(2)
    adapter = TinyGlobalAdapter().eval()
    inputs = torch.randn(2, 6, adapter.hidden_size)
    mask = torch.ones(2, 6, dtype=torch.long)
    if with_padding:
        mask[1, :2] = 0
    full = adapter(inputs, mask).hidden_states
    prefill = adapter(inputs[:, :3], mask[:, :3], use_cache=True)
    incremental = adapter(
        inputs[:, 3:], mask, past_key_values=prefill.past_key_values, use_cache=True
    )
    torch.testing.assert_close(incremental.hidden_states, full[:, 3:], atol=1e-5, rtol=1e-5)
    assert incremental.past_key_values.get_seq_length() == 6


def test_freezing_preserves_input_gradients_and_unfreeze_updates_weights():
    torch.manual_seed(3)
    adapter = TinyGlobalAdapter()
    adapter.freeze()
    adapter.train()
    assert not adapter.model.training
    inputs = torch.randn(1, 4, adapter.hidden_size, requires_grad=True)
    adapter(inputs).hidden_states.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    adapter.unfreeze()
    adapter.train()
    assert adapter.model.training
    parameter = adapter.backbone.layers[0].self_attn.q_proj.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    adapter(inputs.detach()).hidden_states[..., 0].sum().backward()
    optimizer.step()
    assert not torch.equal(before, parameter)


def test_teacher_matches_source_lm_and_restores_mode():
    torch.manual_seed(4)
    adapter = TinyGlobalAdapter()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output = adapter.teacher_forward(input_ids)
    assert adapter.model.training
    assert not output.logits.requires_grad
    assert output.hidden_states.shape == (1, 4, adapter.hidden_size)
    adapter.eval()
    with torch.no_grad():
        expected = adapter.model(input_ids=input_ids).logits
    torch.testing.assert_close(output.logits, expected)


def test_rejects_other_architectures_and_bad_shapes():
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    with pytest.raises(CompatibilityError):
        QwenGlobalAdapter(llama)
    with pytest.raises(CompatibilityError):
        QwenGlobalAdapter.from_config(llama.config)
    adapter = TinyGlobalAdapter()
    with pytest.raises(ValueError, match="inputs_embeds"):
        adapter(torch.randn(1, 3, 17))
    with pytest.raises(ValueError, match="attention_mask"):
        adapter(torch.randn(1, 3, 32), torch.ones(1, 2))
    with pytest.raises(ValueError, match="position_ids"):
        adapter(torch.randn(1, 3, 32), position_ids=torch.ones(1, 4, dtype=torch.long))


def test_from_config_records_identity_without_modifying_weights():
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=16,
    )
    adapter = QwenGlobalAdapter.from_config(
        config, source_model_id="test", source_revision="commit"
    )
    before = {
        name: parameter.detach().clone() for name, parameter in adapter.model.named_parameters()
    }
    wrapper = QwenGlobalAdapter(adapter.model)
    assert wrapper.hidden_size == 16
    assert adapter.source_revision == "commit"
    assert adapter.hf_config["model_type"] == "qwen3"
    assert adapter.original_parameter_count == sum(p.numel() for p in adapter.parameters())
    assert all(
        torch.equal(before[name], parameter) for name, parameter in wrapper.model.named_parameters()
    )


def test_loads_local_pretrained_safetensors_without_network(tmp_path):
    original = TinyGlobalAdapter().eval()
    original.model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = QwenGlobalAdapter.from_pretrained(str(tmp_path), local_files_only=True).eval()
    inputs = torch.randn(1, 3, original.hidden_size)
    torch.testing.assert_close(loaded(inputs).hidden_states, original(inputs).hidden_states)
    assert loaded.source_model_id == str(tmp_path)
    with pytest.raises(ValueError, match="precision"):
        QwenGlobalAdapter.from_pretrained(str(tmp_path), precision="int8", local_files_only=True)
    with pytest.raises(ValueError, match="fp16"):
        QwenGlobalAdapter.from_pretrained(str(tmp_path), precision="fp16", local_files_only=True)
