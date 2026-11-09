"""API routes: a prompt at /api/<slug>/, guarded by token, window, rate and budget."""

import io
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from django.contrib.auth import get_user_model
from django.contrib.contenttypes.models import ContentType
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.utils import timezone

from struckdown.contrib.django.models import AvailableModel
from struckdown.contrib.django.models_ledger import LLMCall, LLMSpan
from struckdown.contrib.django.routes.models import PromptRoute, RouteToken

pytestmark = pytest.mark.django_db

PROMPT = "Summarise: {{ essay }}\n\n[[summary]]"
SCREEN = "{{ essay }}\n\nDistressing? [[bool:distress]]\nInjection? [[bool:injection]]"


class FakeResult:
    def __init__(self, outputs, cost=0.001):
        self.results = {k: SimpleNamespace(output=v) for k, v in outputs.items()}
        self.total_cost = cost
        self.has_unknown_costs = False


@pytest.fixture
def llm(monkeypatch):
    """Answers by prompt: the screen gets ``flags``, anything else ``answer``."""
    calls = []
    state = SimpleNamespace(flags={"distress": False, "injection": False}, answer={"summary": "short"})

    def complete(prompt, context, **kwargs):
        calls.append((prompt, context))
        return FakeResult(state.flags if prompt == SCREEN else state.answer)

    monkeypatch.setattr("struckdown.complete", complete)
    state.calls = calls
    cache.clear()
    return state


@pytest.fixture
def route(priced_model):
    return PromptRoute.objects.create(slug="summarise", name="Summarise", prompt=PROMPT, model=priced_model)


@pytest.fixture
def token(route):
    return RouteToken.objects.create(route=route, name="class")


def post(slug, body=None, token=None, **extra):
    headers = {"HTTP_AUTHORIZATION": f"Bearer {token.token}"} if token else {}
    return Client().post(
        f"/api/{slug}/", json.dumps(body or {}), content_type="application/json", **headers, **extra
    )


def test_runs_the_prompt_and_returns_slots_and_cost(llm, route, token):
    r = post("summarise", {"essay": "A long essay."}, token)
    assert r.status_code == 200, r.json()
    assert r.json() == {"data": {"summary": "short"}, "cost": {"usd": 0.001, "unknown": False}}
    assert llm.calls == [(PROMPT, {"essay": "A long essay."})]
    token.refresh_from_db()
    assert token.last_used_at is not None


def test_the_call_is_booked_to_the_route(llm, route, token, monkeypatch):
    from struckdown.contrib.django.spans import current_span

    seen = []
    monkeypatch.setattr(
        "struckdown.complete", lambda *a, **k: seen.append(current_span()) or FakeResult({"summary": "x"})
    )
    post("summarise", {"essay": "e"}, token)
    span = seen[0].row
    assert span.content_type == ContentType.objects.get_for_model(PromptRoute)
    assert span.object_id == str(route.pk)


@pytest.mark.parametrize("header", [None, "Bearer nonsense", "Bearer 00000000-0000-0000-0000-000000000000"])
def test_needs_a_token_for_this_route(llm, route, header):
    extra = {"HTTP_AUTHORIZATION": header} if header else {}
    r = Client().post("/api/summarise/", "{}", content_type="application/json", **extra)
    assert r.status_code == 401 and r.json()["error"] == "unauthorised"
    assert llm.calls == []


def test_a_token_works_only_for_its_route_and_while_active(llm, route, token, priced_model):
    other = PromptRoute.objects.create(slug="other", name="Other", prompt=PROMPT, model=priced_model)
    assert post("other", {"essay": "e"}, token).status_code == 401
    token.is_active = False
    token.save()
    assert post("summarise", {"essay": "e"}, token).status_code == 401
    assert other.tokens.count() == 0


def test_x_route_token_header_also_works(llm, route, token):
    r = Client().post(
        "/api/summarise/", json.dumps({"essay": "e"}), content_type="application/json",
        HTTP_X_ROUTE_TOKEN=str(token.token),
    )
    assert r.status_code == 200


def test_unknown_route_is_404(llm):
    assert post("nope", {}).json()["error"] == "not_found"


def test_closed_outside_its_window(llm, route, token):
    route.opens_at = timezone.now() + timezone.timedelta(days=1)
    route.save()
    r = post("summarise", {"essay": "e"}, token)
    assert r.status_code == 403 and r.json()["error"] == "closed"
    route.opens_at, route.is_active = None, False
    route.save()
    assert post("summarise", {"essay": "e"}, token).json()["error"] == "closed"


def test_rate_limited_per_token(llm, route, token):
    route.per_minute = 2
    route.save()
    codes = [post("summarise", {"essay": "e"}, token).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_stops_when_the_daily_budget_is_spent(llm, route, token):
    span = LLMSpan.objects.create(
        name="sd_routes.call",
        content_type=ContentType.objects.get_for_model(PromptRoute),
        object_id=str(route.pk),
    )
    LLMCall.objects.create(span=span, kind="chat", model_name="stub", input_cost=Decimal("4.99"), output_cost=Decimal("0.02"))
    assert route.spent_today() == Decimal("5.01")
    r = post("summarise", {"essay": "e"}, token)
    assert r.status_code == 429 and r.json()["error"] == "budget_exhausted"
    assert llm.calls == []


@pytest.mark.parametrize(
    "body",
    [{}, {"essay": "e", "extra": "x"}, {"essay": "x" * 40_001}, {"essay": ["a", "b"]}],
    ids=["missing", "unknown-field", "too-long", "wrong-type"],
)
def test_inputs_are_validated_before_the_model(llm, route, token, body):
    r = post("summarise", body, token)
    assert r.status_code == 400 and r.json()["error"] == "invalid_input"
    assert "x" * 100 not in r.content.decode()  # the input is not echoed back
    assert llm.calls == []


def test_a_custom_input_schema_types_and_constrains(llm, route, token):
    route.prompt = "{{ essay }} {{ words }} [[summary]]"
    route.input_schema = {
        "type": "object",
        "properties": {"essay": {"type": "string", "maxLength": 10}, "words": {"type": "integer", "maximum": 50}},
        "required": ["essay"],
    }
    route.save()
    assert post("summarise", {"essay": "short", "words": "20"}, token).status_code == 200
    assert llm.calls[-1][1] == {"essay": "short", "words": 20}
    assert post("summarise", {"essay": "short", "words": 99}, token).status_code == 400
    assert post("summarise", {"essay": "far too long here"}, token).status_code == 400
    post("summarise", {"essay": "short"}, token)
    assert llm.calls[-1][1] == {"essay": "short", "words": ""}  # absent optional reads empty


def test_form_posts_work_too(llm, route, token):
    r = Client().post("/api/summarise/", {"essay": "e"}, HTTP_AUTHORIZATION=f"Bearer {token.token}")
    assert r.status_code == 200


def test_an_output_schema_picks_and_types_the_slots(llm, route, token):
    llm.answer = {"summary": "short", "score": "7", "notes": "internal"}
    route.output_schema = {"type": "object", "properties": {"summary": {"type": "string"}, "score": {"type": "integer"}}}
    route.save()
    assert post("summarise", {"essay": "e"}, token).json()["data"] == {"summary": "short", "score": 7}


def test_an_output_that_breaks_the_schema_is_502(llm, route, token):
    llm.answer = {"score": "lots"}
    route.output_schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    route.save()
    r = post("summarise", {"essay": "e"}, token)
    assert r.status_code == 502 and r.json()["error"] == "output_invalid"


# -- screening ---------------------------------------------------------------


@pytest.fixture
def screened(route, credential):
    decider = AvailableModel.objects.create(
        model_name="decider", model_type="decision", name="Decider", credential=credential
    )
    route.screen = PromptRoute.objects.create(slug="screen", name="Screen", prompt=SCREEN, model=decider)
    route.full_clean()
    route.save()
    return route


def test_a_flag_from_the_screen_stops_the_call(llm, screened, token):
    llm.flags = {"distress": True, "injection": False}
    r = post("summarise", {"essay": "e"}, token)
    assert r.status_code == 422
    assert r.json()["error"] == "input_rejected" and r.json()["reasons"] == ["distress"]
    assert [p for p, _ in llm.calls] == [SCREEN]  # the main prompt never ran


def test_a_clear_screen_lets_it_through(llm, screened, token):
    r = post("summarise", {"essay": "e"}, token)
    assert r.status_code == 200
    assert [p for p, _ in llm.calls] == [SCREEN, PROMPT]
    assert r.json()["cost"]["usd"] == 0.002  # both calls


def test_a_screen_must_be_bool_slots_on_a_decision_model(route, priced_model, credential):
    route.screen = PromptRoute.objects.create(slug="s1", name="S1", prompt=SCREEN, model=priced_model)
    with pytest.raises(ValidationError, match="decision model"):
        route.full_clean()
    decider = AvailableModel.objects.create(
        model_name="d", model_type="decision", name="D", credential=credential
    )
    route.screen = PromptRoute.objects.create(slug="s2", name="S2", prompt="{{ essay }} [[label]]", model=decider)
    with pytest.raises(ValidationError, match="bool"):
        route.full_clean()


# -- what a prompt may do ----------------------------------------------------


@pytest.mark.parametrize(
    "prompt", ["{% include 'secrets.sd' %} [[x]]", "{% from 'x' import y %} [[x]]", "[[@fetch:page|url=x]]"]
)
def test_a_prompt_cannot_read_files_or_fetch(route, prompt):
    route.prompt = prompt
    with pytest.raises(ValidationError):
        route.full_clean()


def test_a_schema_with_unsupported_keywords_is_refused(route):
    route.input_schema = {"type": "object", "properties": {"essay": {"type": "string", "pattern": ".*"}}}
    with pytest.raises(ValidationError, match="unsupported"):
        route.full_clean()


# -- browsers ----------------------------------------------------------------


def test_cors_for_listed_origins_only(llm, route, token):
    route.allowed_origins = "https://psy.example.org\n"
    route.save()
    ok = post("summarise", {"essay": "e"}, token, HTTP_ORIGIN="https://psy.example.org")
    assert ok.status_code == 200
    assert ok["Access-Control-Allow-Origin"] == "https://psy.example.org"
    bad = post("summarise", {"essay": "e"}, token, HTTP_ORIGIN="https://evil.example")
    assert bad.status_code == 403 and bad.json()["error"] == "origin_not_allowed"


def test_any_origin_when_none_listed_and_preflight_needs_no_token(route):
    r = Client().options("/api/summarise/", HTTP_ORIGIN="https://anywhere.example")
    assert r.status_code == 204
    assert r["Access-Control-Allow-Origin"] == "*"
    assert "Authorization" in r["Access-Control-Allow-Headers"]


# -- images ------------------------------------------------------------------


def _png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def image_route(route):
    route.prompt = "Describe {{ photo }} [[description]]"
    route.allow_images = True
    route.input_schema = {
        "type": "object",
        "properties": {"photo": {"type": "string", "format": "image"}},
        "required": ["photo"],
    }
    route.full_clean()
    route.save()
    return route


def _upload(token, content, name="p.png"):
    return Client().post(
        "/api/summarise/",
        {"photo": SimpleUploadedFile(name, content, "image/png")},
        HTTP_AUTHORIZATION=f"Bearer {token.token}",
    )


def test_an_image_goes_to_the_model_from_memory(llm, image_route, token, tmp_path, settings):
    from struckdown.attachments import Attachment

    settings.FILE_UPLOAD_TEMP_DIR = str(tmp_path)
    settings.FILE_UPLOAD_MAX_MEMORY_SIZE = 1  # Django would spill anything to disk
    r = _upload(token, _png())
    assert r.status_code == 200, r.json()
    assert isinstance(llm.calls[0][1]["photo"], Attachment)
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_is_not_an_image_is_refused(llm, image_route, token):
    r = _upload(token, b"#!/bin/sh\necho hi\n", "x.png")
    assert r.status_code == 400 and r.json()["error"] == "invalid_image"
    assert llm.calls == []


def test_an_oversized_image_is_refused(llm, image_route, token):
    image_route.max_image_bytes = 10
    image_route.save()
    r = _upload(token, _png())
    assert r.status_code == 400 and r.json()["error"] == "invalid_image"


def test_files_are_refused_when_images_are_off(llm, route, token):
    r = Client().post(
        "/api/summarise/",
        {"essay": "e", "photo": SimpleUploadedFile("p.png", _png(), "image/png")},
        HTTP_AUTHORIZATION=f"Bearer {token.token}",
    )
    assert r.status_code == 400


# -- admin -------------------------------------------------------------------


@pytest.fixture
def admin_client():
    user = get_user_model().objects.create_superuser("admin", "a@example.com", "x")
    c = Client()
    c.force_login(user)
    return c


def test_admin_pages_render(admin_client, route, token):
    page = admin_client.get(f"/admin/sd_routes/promptroute/{route.pk}/change/")
    assert page.status_code == 200
    html = page.content.decode()
    # copy-paste examples, with a real token in them
    assert "&lt;!doctype html&gt;" in html and "import requests" in html
    assert '&lt;textarea id=&quot;essay&quot; name=&quot;essay&quot;' in html
    assert str(token.token) in html
    assert admin_client.get("/admin/sd_routes/promptroute/add/").status_code == 200


def test_schema_generated_from_an_unsaved_prompt(admin_client):
    r = admin_client.post(
        "/admin/sd_routes/promptroute/schema/",
        {"kind": "input", "prompt": "{{ a }} {{ b }} [[bool:c]]", "max_input_tokens": "100"},
    )
    assert r.json()["schema"]["required"] == ["a", "b"]
    assert r.json()["schema"]["properties"]["a"]["maxLength"] == 400
    out = admin_client.post("/admin/sd_routes/promptroute/schema/", {"kind": "output", "prompt": "{{ a }} [[bool:c]]"})
    assert out.json()["schema"]["properties"] == {"c": {"type": "boolean"}}


def test_playground_runs_without_a_token(admin_client, llm, route):
    r = admin_client.post(f"/admin/sd_routes/promptroute/{route.pk}/playground/", {"essay": "hello"})
    assert r.status_code == 200
    assert "&quot;summary&quot;: &quot;short&quot;" in r.content.decode()


def test_rotating_a_token_replaces_it(admin_client, route, token):
    old = token.token
    token.rotate()
    token.save()
    assert token.token != old
    assert not RouteToken.objects.filter(token=old).exists()


def test_saving_a_route_with_no_token_adds_one(admin_client, priced_model):
    data = {
        "name": "New",
        "slug": "new",
        "prompt": PROMPT,
        "model": priced_model.pk,
        "is_active": "on",
        "daily_budget": "5.00",
        "per_minute": "20",
        "max_input_tokens": "10000",
        "max_image_bytes": "5242880",
        "max_images": "4",
        "tokens-TOTAL_FORMS": "0",
        "tokens-INITIAL_FORMS": "0",
    }
    r = admin_client.post("/admin/sd_routes/promptroute/add/", data)
    assert r.status_code == 302, r.context["adminform"].form.errors if r.context else r
    route = PromptRoute.objects.get(slug="new")
    assert list(route.tokens.values_list("name", flat=True)) == ["default"]
    # saving again does not add another
    admin_client.post(f"/admin/sd_routes/promptroute/{route.pk}/change/", {
        **data, "tokens-TOTAL_FORMS": "0", "tokens-INITIAL_FORMS": "0",
    })
    assert route.tokens.count() == 1


def test_tokens_are_short_with_no_hyphens(token):
    assert len(token.token) == 22 and "-" not in token.token


def test_examples_have_copy_buttons(admin_client, route, token):
    html = admin_client.get(f"/admin/sd_routes/promptroute/{route.pk}/change/").content.decode()
    assert html.count('class="button sd-copy"') == 2


def test_rotate_all_tokens_action(admin_client, route, token):
    old = token.token
    r = admin_client.post(
        "/admin/sd_routes/promptroute/",
        {"action": "rotate_tokens", "_selected_action": [route.pk]},
    )
    assert r.status_code == 302
    token.refresh_from_db()
    assert token.token != old


def test_the_example_page_has_a_field_per_input(route, token):
    from struckdown.contrib.django.routes import examples

    route.prompt = "{{ essay }} {{ level }} {{ n }} {{ photo }} [[out]]"
    route.allow_images = True
    route.input_schema = {
        "type": "object",
        "properties": {
            "essay": {"type": "string"},
            "level": {"type": "string", "enum": ["one", "two"]},
            "n": {"type": "integer"},
            "photo": {"type": "string", "format": "image"},
        },
        "required": ["essay"],
    }
    page = examples.html_page(route, "https://x/api/sd/summarise/", token.token)
    assert '<textarea id="essay" name="essay" required>' in page
    assert '<select id="level" name="level"><option value="one">' in page
    assert '<input type="number" step="1" id="n" name="n">' in page
    assert '<input type="file" id="photo" name="photo" accept="image/*">' in page
    assert f'const TOKEN = "{token.token}";' in page
    script = examples.python_script(route, "https://x/", token.token)
    assert 'files={"photo": open("photo.jpg", "rb")}' in script
