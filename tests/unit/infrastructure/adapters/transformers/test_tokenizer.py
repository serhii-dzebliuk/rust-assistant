import sys

import pytest

from rust_assistant.infrastructure.adapters.transformers.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.unit


class FakeAutoTokenizer:
    tokenizer = None
    model_names = []

    @classmethod
    def from_pretrained(cls, model_name):
        cls.model_names.append(model_name)
        return cls.tokenizer


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return list(range(len(text.split())))


def test_transformers_tokenizer_counts_text_tokens(monkeypatch):
    tokenizer = FakeTokenizer()
    FakeAutoTokenizer.tokenizer = tokenizer
    FakeAutoTokenizer.model_names = []
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("FakeTransformersModule", (), {"AutoTokenizer": FakeAutoTokenizer}),
    )

    adapter = TransformersTokenizer(" fake-embedding-model ")

    assert adapter.model_name == "fake-embedding-model"
    assert adapter.count_tokens("Returns a Future.") == 3
    assert FakeAutoTokenizer.model_names == ["fake-embedding-model"]
    assert tokenizer.calls == [("Returns a Future.", False)]


def test_transformers_tokenizer_rejects_blank_model_name():
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        TransformersTokenizer(" ")
