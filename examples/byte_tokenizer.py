"""Run with: uv run python examples/byte_tokenizer.py"""

import json

from bytepft import ByteTokenizer
from bytepft.tokenization.byte_tokenizer import IncrementalUTF8Decoder


def main() -> None:
    tokenizer = ByteTokenizer()
    text = "Hello, दुनिया 🌍"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    raw = bytes(range(256))
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(raw)) == raw
    decoder = IncrementalUTF8Decoder()
    streamed = "".join(decoder.feed([value]) for value in ids) + decoder.finish()
    assert streamed == text
    print(
        json.dumps({"text": text, "byte_ids": ids, "streamed_text": streamed}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
