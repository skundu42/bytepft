import json

import pytest
import torch

from bytepft.training.datasets import (
    ByteDataset,
    collate_documents,
    deterministic_split,
    load_texts,
    source_boundary_targets,
)


def test_dataset_loading_preserves_documents_and_sorted_files(tmp_path):
    (tmp_path / "b.txt").write_text("two\nlines", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"body": "नमस्ते"}, ensure_ascii=False) + "\n" + json.dumps({"body": ""}) + "\n",
        encoding="utf-8",
    )
    assert load_texts(tmp_path, text_field="body") == ["नमस्ते", "", "two\nlines"]
    assert load_texts(["one", "two"]) == ["one", "two"]


def test_dataset_loading_reports_bad_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": 42}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_texts(path)
    path.write_text("{broken}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_texts(path)
    with pytest.raises(FileNotFoundError):
        load_texts(tmp_path / "absent.txt")


def test_text_files_preserve_crlf_bytes(tmp_path):
    path = tmp_path / "windows.txt"
    path.write_bytes(b"first\r\nsecond\r\n")
    assert load_texts(path) == ["first\r\nsecond\r\n"]


def test_explicit_hf_loading_and_missing_extra(monkeypatch):
    from types import SimpleNamespace

    from bytepft.training import datasets

    def fake_import(name):
        assert name == "datasets"
        return SimpleNamespace(load_dataset=lambda name, split: {"body": [name, split]})

    monkeypatch.setattr(datasets, "import_module", fake_import)
    assert load_texts(hf_dataset="fixture", text_field="body", split="test") == ["fixture", "test"]
    with pytest.raises(ValueError, match="either"):
        load_texts(["local"], hf_dataset="fixture")

    def missing_extra(name):
        raise ImportError("No module named datasets")

    monkeypatch.setattr(datasets, "import_module", missing_extra)
    with pytest.raises(ImportError, match="bytepft"):
        load_texts(hf_dataset="fixture")


def test_split_is_deterministic_disjoint_and_document_preserving():
    docs = [str(i) for i in range(10)]
    first = deterministic_split(docs, 0.2, seed=17)
    assert first == deterministic_split(docs, 0.2, seed=17)
    train, validation = first
    assert len(train) == 8 and len(validation) == 2
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(docs)
    assert deterministic_split(["only"], 0.5) == (["only"], [])


def test_collation_bos_eos_shift_padding_and_text_alignment():
    dataset = ByteDataset(["ab", "🌍x", ""], max_bytes=3)
    assert dataset.texts == ["ab", "", ""]
    batch = collate_documents([dataset[0], dataset[1], dataset[2]])
    assert batch["input_ids"].tolist() == [
        [257, 97, 98, 258],
        [257, 258, 256, 256],
        [257, 258, 256, 256],
    ]
    assert batch["labels"].tolist() == [
        [97, 98, 258, -100],
        [258, -100, -100, -100],
        [258, -100, -100, -100],
    ]
    assert batch["attention_mask"].tolist() == [
        [True] * 4,
        [True, True, False, False],
        [True, True, False, False],
    ]
    assert batch["texts"] == ["ab", "", ""]
    assert batch["byte_counts"] == [2, 0, 0]


def test_dataset_rejects_invalid_limits_and_documents():
    with pytest.raises(ValueError, match="positive integer"):
        ByteDataset(["abc"], max_bytes=0)
    with pytest.raises(ValueError, match="string"):
        ByteDataset([42])
    with pytest.raises(ValueError, match="empty batch"):
        collate_documents([])


def test_source_boundary_targets_only_supervise_raw_bytes():
    ids = torch.tensor([[257, 65, 66, 67, 258], [257, 68, 258, 256, 256]])
    targets, mask = source_boundary_targets(ids, [[1, 2], [0]])
    assert targets.tolist() == [[0, 0, 1, 1, 0], [0, 1, 0, 0, 0]]
    assert mask.tolist() == [[False, True, True, True, False], [False, True, False, False, False]]
    with pytest.raises(ValueError, match="raw"):
        source_boundary_targets(ids, [[3], [0]])
