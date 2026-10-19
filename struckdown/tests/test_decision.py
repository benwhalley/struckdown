"""Decision models: systemone payloads, per-slot routing, <together> batching (HTTP mocked)."""

import logging
import uuid
from types import SimpleNamespace

import httpx
import pytest

from struckdown import (
    LLM,
    LLMCredentials,
    ModelSpec,
    complete,
    decision,
    usage_tracking,
)
from struckdown import llm as llm_module
from struckdown.decision import split_state_and_instructions
from struckdown.model_spec import ModelRegistry

FAKE_ANSWERS = {
    "choice": lambda q: {
        "type": "choice",
        "choice": list(q["criteria"])[1],
        "probabilities": {k: (0.9 if i == 1 else 0.1 / (len(q["criteria"]) - 1))
                          for i, k in enumerate(q["criteria"])},
        "confidence": 0.8,
    },
    "noul": lambda q: {"type": "noul", "noul": 0.82},
    "score": lambda q: {
        "type": "score",
        "score": 2.3,
        "legend": {str(i): lvl for i, lvl in enumerate(q["criteria"])},
        "probabilities": {"0": 0.0, "1": 0.1, "2": 0.5, "3": 0.4},
        "confidence": 0.6,
    },
}

DECISION = ModelSpec(model_name="azure:Decision1", model_type="decision", alias="decision",
                     api_key="az", base_url="https://x.cognitiveservices.azure.com")
CHAT = ModelSpec(model_name="openai:gpt-4.1-mini", alias="fast", api_key="sk")


@pytest.fixture
def requests_seen(monkeypatch):
    seen = []

    def fake_post(url, json, headers, timeout):
        seen.append({"url": url, "json": json, "headers": headers})
        body = {
            "model": "microsoft-decision-1",
            "answers": {k: FAKE_ANSWERS[q["type"]](q) for k, q in json["questions"].items()},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(decision.httpx, "post", fake_post)
    return seen


@pytest.fixture
def chat_calls(monkeypatch):
    """Stand in for chat-model calls: record the model and answer from the return type."""
    calls = []

    def fake_cached(messages, model_name, return_type, **kw):
        llm_module._cache_miss_marker.set(True)
        if kw["llm"].model_type == "decision":
            return decision.call_decision_model(messages, return_type, model_name,
                                                kw["credentials"])
        calls.append(model_name)
        if "triggered" in return_type.model_fields:
            return {"triggered": False, "reason": "fine"}, {}
        return {"response": "chat says hi"}, {}

    monkeypatch.setattr(llm_module, "_call_llm_cached", fake_cached)
    return calls


def run(template, model_name="openai:gpt-4.1-mini", credentials=None, model_type="llm", **kw):
    # a unique line in the prompt defeats the cache between runs
    return complete(
        f"Ref {uuid.uuid4()}\n\n{template}",
        model=LLM(model_name=model_name, model_type=model_type),
        credentials=credentials or LLMCredentials(api_key="k"),
        **kw,
    )


def test_split_state_and_instructions():
    msgs = [{"role": "user", "content": "Ticket: the API 500s.\n\nWhich team should handle it?"}]
    assert split_state_and_instructions(msgs) == (
        "Ticket: the API 500s.",
        "Which team should handle it?",
    )


def test_model_type_is_explicit_not_inferred_from_name():
    assert ModelSpec(model_name="azure:Decision1").model_type == "llm"
    assert DECISION.as_llm().model_type == "decision"
    assert CHAT.as_llm().model_type == "llm"


def test_unsupported_provider_for_decision_model(requests_seen):
    with pytest.raises(Exception, match="No decision-model endpoint for provider 'anthropic'"):
        run("Q? [[bool:x]]", "anthropic:claude", model_type="decision")


def test_registry_match():
    reg = ModelRegistry.from_specs([DECISION, CHAT])
    assert reg.match("decision") == (DECISION, True)
    assert reg.match("gpt-4.1-mini") == (CHAT, True)
    assert reg.match("Decision") == (DECISION, False)
    assert reg.resolve_decision() == DECISION
    with pytest.raises(ValueError, match="No model matches"):
        reg.match("claude")
    with pytest.raises(ValueError, match="several"):
        reg.match("i")


def test_pick_on_decision_default_model(requests_seen):
    res = run(
        "Ticket: the API returns 500 on every call.\n\n"
        "Which team should handle this? [[pick:team|billing,engineering,support]]",
        "typesafe:jev-latest",
        model_type="decision",
    )
    req = requests_seen[0]
    assert req["url"] == "https://api.typesafe.ai/v1/systemone"
    assert req["headers"]["Authorization"] == "Bearer k"
    q = req["json"]["questions"]["q"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"billing", "engineering", "support"}
    assert "500" in req["json"]["state"]
    assert res["team"] == "engineering"
    assert res.results["team"].decision["probabilities"]["engineering"] == 0.9
    assert res.results["team"].decision["calibrated"] is True


def test_score_type(requests_seen):
    res = run("Comment: ...\n\nHow concerning is this? [[score:level|none,low,moderate,high]]",
              "azure:decision-1",
              LLMCredentials(api_key="az", base_url="https://x.cognitiveservices.azure.com/"),
              model_type="decision")
    req = requests_seen[0]
    assert req["url"] == "https://x.cognitiveservices.azure.com/providers/microsoft/v1/systemone"
    assert req["headers"]["api-key"] == "az"
    assert req["json"]["model"] == "decision-1"
    assert req["json"]["questions"]["q"]["criteria"] == ["none", "low", "moderate", "high"]
    assert res["level"] == 2.3


def test_free_text_slot_rejected_by_decision_model(requests_seen):
    with pytest.raises(Exception, match="decision model can only answer"):
        run("Say hi [[greeting]]", "typesafe:jev-latest", model_type="decision")


@pytest.mark.parametrize("base", ["https://proxy.example", "https://proxy.example/v1/"])
def test_unprefixed_gateway_eg_litellm_proxy(requests_seen, base):
    run("Text: hi\n\nIs this a greeting? [[bool:greeting]]", "decision-1",
        LLMCredentials(api_key="sk-proxy", base_url=base), model_type="decision")
    req = requests_seen[0]
    assert req["url"] == "https://proxy.example/v1/systemone"
    assert req["headers"]["Authorization"] == "Bearer sk-proxy"
    assert req["json"]["model"] == "decision-1"


def test_decide_without_decision_model_fails(chat_calls):
    with pytest.raises(Exception, match="needs a decision model"):
        run("Is this urgent? [[decide:urgent]]")
    with pytest.raises(Exception, match="No decision model available"):
        run("Is this urgent? [[decide:urgent]]", registry=[CHAT])
    assert chat_calls == []


def test_decide_and_mode_decision_route_to_registry_model(requests_seen, chat_calls):
    res = run(
        "Comment: I can't cope.\n\n"
        "Write a reply. [[reply]]\n\n"
        "Is there risk of harm? [[decide:risk]]\n\n"
        "Topic? [[pick:topic|teaching,wellbeing,other,mode=decision]]\n\n"
        "Which one? [[decide:which|a,b,c]]",
        registry=[DECISION, CHAT],
    )
    assert chat_calls == ["openai:gpt-4.1-mini"]
    assert len(requests_seen) == 3
    assert {r["headers"]["api-key"] for r in requests_seen} == {"az"}
    assert [r["json"]["questions"]["q"]["type"] for r in requests_seen] == [
        "noul", "choice", "choice"]
    assert res["reply"] == "chat says hi"
    assert res["risk"] is True
    assert res["topic"] == "wellbeing"
    assert res["which"] == "b"


def test_model_alias_and_fuzzy_warning(requests_seen, chat_calls, caplog):
    run("Q? [[bool:a|model=decision]]", registry=[DECISION, CHAT])
    assert len(requests_seen) == 1
    assert "FUZZY" not in caplog.text

    with caplog.at_level(logging.WARNING):
        run("Q? [[bool:b|model=Decisi]]", registry=[DECISION, CHAT])
    assert len(requests_seen) == 2
    assert "FUZZY MODEL MATCH" in caplog.text

    with pytest.raises(Exception, match="No model matches"):
        run("Q? [[bool:c|model=claude]]", registry=[DECISION, CHAT])


def test_model_override_without_registry_uses_the_name_as_given(chat_calls):
    run('Hi [[x|model="openai:gpt-5"]]')
    run('Hi [[y|model="gpt-4o-mini"]]')
    assert chat_calls == ["openai:gpt-5", "gpt-4o-mini"]


def test_mode_decision_on_non_decision_model_fails(chat_calls):
    with pytest.raises(Exception, match="isn't one"):
        run("Q? [[bool:x|mode=decision,model=fast]]", registry=[DECISION, CHAT])


def test_halt_on_decision_model(requests_seen, chat_calls):
    res = run("Message: hello\n\nIs this abusive? [[halt:abuse|mode=decision,when=false]]",
              registry=[DECISION, CHAT])
    q = requests_seen[0]["json"]["questions"]["q"]
    assert q["type"] == "noul"
    # noul 0.82 -> triggered; when=false halts on *not* triggered, so the run continues
    assert res.results["abuse"].decision["noul"] == 0.82


def test_together_batches_decision_slots(requests_seen, chat_calls):
    res = run(
        "Comment: the deadlines all landed in one week.\n\n"
        "<together>\n"
        "Is the student at risk? [[decide:risk]]\n\n"
        "Topic? [[decide:topic|teaching,workload,other]]\n\n"
        "Urgency? [[score:urgency|none,low,high,now,mode=decision]]\n\n"
        "Summarise. [[summary]]\n"
        "</together>",
        registry=[DECISION, CHAT],
    )
    assert len(requests_seen) == 1
    sent = requests_seen[0]["json"]
    assert list(sent["questions"]) == ["risk", "topic", "urgency"]
    assert sent["questions"]["topic"]["instructions"] == "Topic?"
    assert "deadlines" in sent["state"]
    assert "Topic?" not in sent["state"]
    assert chat_calls == ["openai:gpt-4.1-mini"]
    assert (res["risk"], res["topic"], res["urgency"]) == (True, "workload", 2.3)
    costs = [res.results[k].completion["_hidden_params"]["response_cost"]
             for k in ("risk", "topic", "urgency")]
    assert costs[0] > 0 and costs[1:] == [0, 0]


def test_chat_answers_get_uncalibrated_decision(monkeypatch):
    monkeypatch.setattr(
        llm_module, "_call_llm_cached",
        lambda messages, model_name, return_type, **kw: ({"response": "b"}, {}),
    )
    res = run("Pick [[pick:x|a,b,c]]")
    assert res.results["x"].decision == {
        "type": "choice", "calibrated": False, "choice": "b",
        "probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}, "confidence": None,
    }


def test_registry_pricing_applies_per_model():
    llm_module.set_registry_pricing([
        ModelSpec(model_name="openai:cheap", input_cost_per_mtok=1, output_cost_per_mtok=2),
        ModelSpec(model_name="openai:unpriced"),
    ])
    usage = SimpleNamespace(usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0))
    try:
        cost = lambda name: llm_module._calc_cost_from_usage(usage, name).total_cost
        assert cost("cheap") == 1
        llm_module.set_model_pricing(9, 9)
        assert cost("cheap") == 1
        assert cost("other") == 9
    finally:
        llm_module.set_registry_pricing([])
        llm_module.set_model_pricing(None, None)


def test_decision_calls_reach_the_usage_ledger(requests_seen, chat_calls):
    records = []
    with usage_tracking(records.append):
        run(
            "Comment: the deadlines all landed in one week.\n\n"
            "Is it about workload? [[decide:solo]]\n\n"
            "<together>\n"
            "Is the student at risk? [[decide:risk]]\n\n"
            "Topic? [[decide:topic|teaching,workload,other]]\n"
            "</together>",
            registry=[DECISION, CHAT],
        )
    assert len(requests_seen) == 2
    assert [r.slot for r in records] == ["solo", "risk,topic"]
    assert {r.model_name for r in records} == {"azure:Decision1"}
    assert all(r.input_tokens == 120 and r.cost.source == "decision_rate" for r in records)
    assert records[0].cost.total_cost == pytest.approx(120 * 0.042 / 1e6)


def test_rate_limits_are_retried(monkeypatch, chat_calls):
    statuses = iter([429, 503])
    sent = []

    def flaky_post(url, json, headers, timeout):
        sent.append(url)
        status = next(statuses, 200)
        body = {"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 5}}
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(decision.httpx, "post", flaky_post)
    monkeypatch.setattr(decision._post_systemone.func.retry, "sleep", lambda s: None)
    res = run("Q? [[decide:x]]", registry=[DECISION, CHAT])
    assert len(sent) == 3
    assert res["x"] is True


def test_routed_calls_are_booked_to_their_own_model_ref(requests_seen, chat_calls):
    """A host names the run's model with set_model_ref; a slot routed elsewhere must not
    inherit it, or its spend joins back to the wrong row."""
    from struckdown.ledger import get_model_ref, set_model_ref

    records = []
    set_model_ref("7")
    try:
        with usage_tracking(records.append):
            run(
                "Comment: the deadlines all landed in one week.\n\n"
                "Is it about workload? [[decide:solo]]\n\n"
                "Urgency? [[score:urgency|none,low,high,now,mode=decision]]\n\n"
                "Summarise. [[summary]]\n\n"
                "<together>\n"
                "Is the student at risk? [[decide:risk]]\n\n"
                "Topic? [[decide:topic|teaching,workload,other]]\n"
                "</together>\n\n"
                "Again. [[anon|model=unreffed]]",
                registry=[DECISION.model_copy(update={"model_ref": "42"}), CHAT,
                          ModelSpec(model_name="openai:unreffed", alias="unreffed")],
            )
        assert get_model_ref() == "7"
    finally:
        set_model_ref(None)
    refs = {r.slot: r.model_ref for r in records}
    assert refs == {"solo": "42", "urgency": "42", "summary": "7", "risk,topic": "42",
                    "anon": None}
