import pytest
import torch

from bytepft.tokenization import ByteTokenizer, IncrementalUTF8Decoder


@pytest.mark.parametrize("text", ["", "ASCII\x00\n", "हिन्दी 日本語", "🌱👩🏽‍💻", "é e\u0301", "𝄞"])
def test_unicode_roundtrip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode(text) == list(text.encode("utf-8"))


def test_raw_byte_roundtrip_and_strict_text_decoding():
    tokenizer = ByteTokenizer()
    data = bytes(range(256))
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode(tokenizer.encode_bytes(data))
    with pytest.raises(TypeError):
        tokenizer.encode(data)
    with pytest.raises(TypeError):
        tokenizer.encode_bytes("hello")


def test_special_tokens_are_not_bytes():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("A", add_special_tokens=True)
    assert ids == [257, 65, 258]
    assert tokenizer.decode(ids + [256]) == "A"
    with pytest.raises(ValueError, match="Control"):
        tokenizer.decode(ids, skip_special_tokens=False)
    with pytest.raises(ValueError, match="Unknown"):
        tokenizer.decode([259])
    with pytest.raises(ValueError, match="Unknown"):
        tokenizer.decode([-1])
    with pytest.raises(TypeError, match="integers"):
        tokenizer.decode([1.5])


def test_padding_and_tensor_interface():
    tokenizer = ByteTokenizer()
    batch = tokenizer(["a", "💡", ""], return_tensors="pt")
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].shape == (3, 4)
    assert batch["attention_mask"].tolist() == [[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
    assert tokenizer.batch_decode(batch["input_ids"]) == ["a", "💡", ""]
    assert tokenizer("a")["input_ids"] == [97]
    assert tokenizer("", return_tensors="pt")["input_ids"].shape == (1, 0)
    assert tokenizer([], return_tensors="pt")["input_ids"].shape == (0, 0)
    with pytest.raises(ValueError, match="equal lengths"):
        tokenizer(["a", "bb"], padding=False, return_tensors="pt")


def test_explicit_truncation_and_max_length():
    tokenizer = ByteTokenizer()
    result = tokenizer(["abcd", "a"], max_length=2, truncation=True, padding="max_length")
    assert result["input_ids"] == [[97, 98], [97, 256]]
    with pytest.raises(ValueError, match="exceeds"):
        tokenizer("abcd", max_length=2, padding="max_length")
    with pytest.raises(ValueError, match="requires max_length"):
        tokenizer("a", padding="max_length")
    for invalid_length in (1.5, True, -1):
        with pytest.raises(ValueError, match="nonnegative integer"):
            tokenizer("a", max_length=invalid_length)


def test_incremental_decode_buffers_multibyte_suffix():
    decoder = IncrementalUTF8Decoder()
    encoded = "a💡é".encode()
    output = ""
    for value in encoded:
        output += decoder.feed([value])
    output += decoder.finish()
    assert output == "a💡é"
    decoder.feed([0xF0])
    assert decoder.pending_bytes == bytes([0xF0])
    with pytest.raises(UnicodeDecodeError):
        decoder.finish()
    decoder.reset()
    assert decoder.feed(b"ok") == "ok"


def test_tokenizer_serialization(tmp_path):
    tokenizer = ByteTokenizer()
    paths = tokenizer.save_pretrained(tmp_path)
    assert len(paths) == 1
    restored = ByteTokenizer.from_pretrained(tmp_path)
    assert restored.encode("你好", add_special_tokens=True) == tokenizer.encode(
        "你好", add_special_tokens=True
    )
    target = tmp_path / "byte_tokenizer.json"
    target.write_text(target.read_text().replace('"pad_token_id": 256', '"pad_token_id": 0'))
    with pytest.raises(ValueError, match="modified byte mapping"):
        ByteTokenizer.from_pretrained(tmp_path)
