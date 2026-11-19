import csv
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel

import src.llm as llm


class Out(BaseModel):
    text: str


class FakeClient:
    """Stands in for OpenAI(): counts calls and returns what the test asks for."""

    def __init__(self):
        self.calls, self.embed_calls, self.answer, self.fail_on = 0, 0, Out(text="hello"), None
        self.responses = SimpleNamespace(parse=self._parse)
        self.embeddings = SimpleNamespace(create=self._embed)

    def _parse(self, **kw):
        self.calls += 1
        if self.fail_on and self.fail_on in kw["input"][1]["content"]:
            raise RuntimeError("boom")
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=2), output_parsed=self.answer)

    def _embed(self, model, input):
        self.embed_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))] * 4) for t in input])


@pytest.fixture
def fake(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "CACHE_DB", tmp_path / "cache.sqlite")
    monkeypatch.setattr(llm, "CALL_LOG", tmp_path / "calls.csv")
    monkeypatch.setattr(llm, "_openai", lambda: client)
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.delenv("LLM_CACHE", raising=False)
    return client


def test_cache_hit_makes_no_second_call_and_is_logged(fake):
    first = llm.call(Out, "system", "user", purpose="t")
    second = llm.call(Out, "system", "user", purpose="t")
    assert first == second and fake.calls == 1
    with open(llm.CALL_LOG) as f:
        rows = list(csv.DictReader(f))
    assert [r["cached"] for r in rows] == ["0", "1"] and rows[0]["input_tokens"] == "10"


def test_another_prompt_or_version_is_not_a_cache_hit(fake):
    llm.call(Out, "system", "user", purpose="t")
    llm.call(Out, "system", "other", purpose="t")
    llm.call(Out, "system", "user", purpose="t", version="v2")
    assert fake.calls == 3


def test_cache_can_be_turned_off(fake, monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "off")
    llm.call(Out, "system", "user", purpose="t")
    llm.call(Out, "system", "user", purpose="t")
    assert fake.calls == 2


def test_refusal_is_retried_once_then_raises(fake):
    fake.answer = None
    with pytest.raises(llm.LLMError):
        llm.call(Out, "system", "user", purpose="t")
    assert fake.calls == 2


def test_call_many_keeps_order_and_returns_errors_in_place(fake):
    fake.fail_on = "bad"
    jobs = [dict(schema=Out, system="s", user=u, purpose="t") for u in ("a", "bad", "c")]
    results = llm.call_many(jobs)
    assert isinstance(results[0], Out) and isinstance(results[1], llm.LLMError) and isinstance(results[2], Out)


def test_embed_caches_per_text(fake):
    first = llm.embed(["one", "three"])
    second = llm.embed(["three", "one"])
    assert first.shape == (2, 4) and fake.embed_calls == 1
    assert np.array_equal(first[0], second[1])


def test_mock_mode_uses_the_registered_fake(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.setitem(llm.MOCKS, "t", lambda system, user: Out(text=user))
    assert llm.call(Out, "s", "from the mock", purpose="t").text == "from the mock"
    assert llm.embed(["a", "b"]).shape == (2, llm.EMBED_DIM)
