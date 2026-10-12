"""Decision models in the Django contrib: the row's type reaches the LLM and the registry."""

from struckdown.contrib.django.models import AvailableModel, ModelSet


def test_decision_row_gives_a_decision_llm(credential):
    am = AvailableModel.objects.create(
        model_name="azure:Decision1", name="Decision", model_type="decision",
        credential=credential, input_cost_per_mtok=0.042, output_cost_per_mtok=0,
    )
    llm, creds = am.get_llm_and_credentials()
    assert llm.model_type == "decision"
    assert creds.base_url == "http://example.invalid/v1"
    assert am.to_spec().model_type == "decision"


def test_chat_row_stays_llm(priced_model):
    assert priced_model.get_llm_and_credentials()[0].model_type == "llm"


def test_model_set_registry_knows_its_decision_model(credential, priced_model):
    decision = AvailableModel.objects.create(
        model_name="azure:Decision1", name="Decision", model_type="decision",
        credential=credential,
    )
    ms = ModelSet.objects.create(name="Set", default_llm=priced_model, default_decision_model=decision)
    ms.available_models.add(priced_model, decision)
    assert ms.to_registry().resolve_decision().model_name == "azure:Decision1"


def test_admin_test_button_asks_a_decision_model_a_decision(credential, monkeypatch):
    import httpx
    from django.contrib import admin

    from struckdown import decision

    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append((url, json))
        answer = {"type": "choice", "choice": "billing", "confidence": 0.9,
                  "probabilities": {"billing": 0.9, "engineering": 0.05, "customer support": 0.05}}
        body = {"answers": {"q": answer}, "usage": {"input_tokens": 40}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(decision.httpx, "post", fake_post)
    am = AvailableModel.objects.create(
        model_name="decision-1", name="Decision", model_type="decision", credential=credential,
    )
    model_admin = admin.site._registry[AvailableModel]
    result = model_admin._test_model(am)
    assert result["response"] == "billing"
    assert result["probabilities"]["billing"] == 0.9
    assert sent[0][0] == "http://example.invalid/v1/systemone"
    assert model_admin._build_test_connection(am)["endpoint"] == sent[0][0]
