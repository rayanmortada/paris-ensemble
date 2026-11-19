"""The real OpenAI SDK against a small local server, so the request format, the answer parsing and the error
messages are tested without a key. This cannot prove that the real API accepts a model name: scripts/smoke_llm.py does."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from openai import OpenAI

import src.llm as llm
import src.schemas as sc
from src.jobs import friendly_error

ALL_MODELS = [sc.ProfileOut, sc.ScoreBatch, sc.MemberReply, sc.MediatorDecision, sc.Explanations, sc.PlannerOut, sc.PersonaText, sc.EnrichBatch]


def response_body(content: dict) -> dict:
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "test", "status": "completed", "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [], "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [content]}],
            "usage": {"input_tokens": 11, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 7,
                      "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 18}}


def text_answer(payload) -> dict:
    return response_body({"type": "output_text", "text": payload if isinstance(payload, str) else json.dumps(payload), "annotations": []})


class Fake:
    """What the server should answer next, and what it received."""

    def __init__(self):
        self.status, self.answer, self.requests = 200, text_answer({"accept": True}), []


@pytest.fixture
def server(tmp_path, monkeypatch):
    fake = Fake()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append((self.path, body))
            if self.path.endswith("/embeddings"):
                vectors = [np.full(llm.EMBED_DIM, len(t), dtype="float32") for t in body["input"]]
                data = [{"object": "embedding", "index": i,
                         "embedding": base64.b64encode(v.tobytes()).decode() if body.get("encoding_format") == "base64" else v.tolist()}
                        for i, v in enumerate(vectors)]
                status, answer = 200, {"object": "list", "model": body["model"], "data": data, "usage": {"prompt_tokens": 1, "total_tokens": 1}}
            else:
                status, answer = fake.status, fake.answer
            raw = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    client = OpenAI(api_key="test-key", base_url=f"http://127.0.0.1:{httpd.server_port}/v1", max_retries=0, timeout=10)
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "CACHE_DB", tmp_path / "cache.sqlite")
    monkeypatch.setattr(llm, "CALL_LOG", tmp_path / "calls.csv")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.setenv("LLM_CACHE", "off")
    yield fake
    httpd.shutdown()


def sample(schema: dict, defs: dict):
    """The smallest valid value for a JSON schema."""
    if "$ref" in schema:
        return sample(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return sample(next(s for s in schema["anyOf"] if s.get("type") != "null"), defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: sample(v, defs) for k, v in schema["properties"].items()}
    return {"array": [], "string": "x", "integer": 1, "number": 1.5, "boolean": True}[kind]


def walk(schema):
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from walk(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from walk(value)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_output_model_goes_through_the_sdk_and_comes_back(server, model):
    schema = model.model_json_schema()
    server.answer = text_answer(sample(schema, schema.get("$defs", {})))
    assert isinstance(llm.call(model, "sys", "usr", purpose="t", model="some-model", effort="medium"), model)

    path, body = server.requests[-1]
    assert path.endswith("/responses") and body["model"] == "some-model" and body["reasoning"] == {"effort": "medium"}
    assert [m["role"] for m in body["input"]] == ["system", "user"]
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    for node in walk(fmt["schema"]):                # what OpenAI strict mode insists on, at every level
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
        assert "default" not in node


def test_a_broken_answer_is_tried_twice_then_raises(server):
    server.answer = text_answer("this is not json")
    with pytest.raises(llm.LLMError):
        llm.call(sc.PersonaText, "s", "u", purpose="t")
    assert len(server.requests) == 2


def test_a_refusal_is_tried_twice_then_raises(server):
    server.answer = response_body({"type": "refusal", "refusal": "I cannot help with that."})
    with pytest.raises(llm.LLMError):
        llm.call(sc.PersonaText, "s", "u", purpose="t")
    assert len(server.requests) == 2


def test_usage_reaches_the_call_log_and_the_totals(server):
    server.answer = text_answer({"wishes": "hello"})
    before = llm.stats()
    assert llm.call(sc.PersonaText, "s", "u", purpose="t").wishes == "hello"
    after = llm.stats()
    assert after["calls"] - before["calls"] == 1 and after["input_tokens"] - before["input_tokens"] == 11
    assert "t," in llm.CALL_LOG.read_text() and ",11,7,0," in llm.CALL_LOG.read_text()


@pytest.mark.parametrize("status, error, words", [
    (401, {"message": "Incorrect API key provided: abc.", "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}, "key is missing or wrong"),
    (429, {"message": "You exceeded your current quota.", "type": "insufficient_quota", "param": None, "code": "insufficient_quota"}, "spending or rate limit"),
    (404, {"message": "The model `gpt-x` does not exist or you do not have access to it.", "type": "invalid_request_error", "param": None, "code": "model_not_found"}, "model name"),
])
def test_real_sdk_errors_become_clear_messages(server, status, error, words):
    server.status, server.answer = status, {"error": error}
    with pytest.raises(llm.LLMError) as caught:
        llm.call(sc.PersonaText, "s", "u", purpose="t")
    assert words in friendly_error(caught.value)


def test_embeddings_through_the_sdk_keep_their_order(server):
    texts = [f"text {'x' * i}" for i in range(230)]           # more than one batch of 100
    vectors = llm.embed(texts)
    calls = [body for path, body in server.requests if path.endswith("/embeddings")]
    assert vectors.shape == (230, llm.EMBED_DIM) and len(calls) == 3
    assert all(vectors[i, 0] == len(t) for i, t in enumerate(texts))
