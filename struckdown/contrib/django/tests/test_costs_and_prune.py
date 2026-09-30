"""The costs page's numbers, its rendering, and the pruning command."""

from datetime import timedelta
from decimal import Decimal
from io import StringIO

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.urls import reverse
from django.utils import timezone

from struckdown.contrib.django import costs, ledger
from struckdown.contrib.django.models import LLMCall, LLMCallPayload, LLMSpan
from struckdown.contrib.django.spans import llm_span
from struckdown.ledger import CostBreakdown, UsagePayload, UsageRecord

pytestmark = pytest.mark.django_db


def _call(feature, cost, *, user=None, cache_hit=False, model_name="stub", tokens=1000):
    breakdown = None if cost is None else CostBreakdown(cost * 0.75, cost * 0.25, source="stored")
    with llm_span(feature, user=user):
        return ledger.write_record(
            UsageRecord(
                kind="chat",
                model_name=model_name,
                input_tokens=tokens,
                cache_read_tokens=tokens // 2,
                output_tokens=100,
                cost=breakdown,
                cache_hit=cache_hit,
            )
        )


@pytest.fixture
def some_calls():
    User = get_user_model()
    a = User.objects.create(username="a")
    b = User.objects.create(username="b")
    _call("ask", 0.01, user=a)
    _call("ask", 0.03, user=b)
    _call("ask", None, user=b)  # unpriced
    _call("ask", 0.0, user=a, cache_hit=True)
    _call("autotag", 0.10, model_name="other")
    ledger.write_record(UsageRecord(kind="embedding", model_name="emb", input_tokens=50))


def test_headline_counts_unpriced_and_cache_hits_apart(some_calls):
    h = costs.headline(30)
    assert h["calls"] == 6
    assert h["cost_total"] == Decimal("0.14")
    assert h["unpriced"] == 2  # one chat, one embedding
    assert h["cache_hits"] == 1
    assert h["users"] == 2
    assert h["cache_share"] == 50


def test_by_feature_groups_on_the_root_span_and_names_the_unattributed(some_calls):
    rows = {r["feature"]: r for r in costs.by_feature(30)}
    assert set(rows) == {"ask", "autotag", costs.UNATTRIBUTED}
    ask = rows["ask"]
    assert ask["calls"] == 4
    assert ask["users"] == 2
    assert ask["cost_total"] == Decimal("0.04")
    # per call divides by priced, non-cached calls only
    assert ask["per_call"] == Decimal("0.02")
    assert ask["unpriced"] == 1
    assert rows[costs.UNATTRIBUTED]["calls"] == 1


def test_per_user_is_counts_and_averages_only(some_calls):
    u = costs.per_user(30)
    assert u["users"] == 2
    assert u["mean"] == pytest.approx(0.02)
    assert u["max"] == pytest.approx(0.03)
    assert "names" not in u


def test_by_month_has_this_month(some_calls):
    rows = costs.by_month()
    assert len(rows) == 1
    assert rows[0]["calls"] == 6


def test_the_window_excludes_old_calls(some_calls):
    LLMCall.objects.update(created_at=timezone.now() - timedelta(days=40))
    assert costs.headline(30)["calls"] == 0
    assert costs.headline(0)["calls"] == 6


class TestCostsPage:
    def test_a_superuser_sees_the_page(self, client, some_calls):
        admin = get_user_model().objects.create_superuser("root", "r@x.invalid", "pw")
        client.force_login(admin)
        response = client.get(reverse("admin:sd_models_llmcosts_changelist"))
        assert response.status_code == 200
        body = response.content.decode()
        assert "By feature" in body
        assert "autotag" in body
        assert "unattributed" in body

    def test_a_staff_user_without_the_permission_is_refused(self, client):
        staff = get_user_model().objects.create_user("s", password="pw", is_staff=True)
        client.force_login(staff)
        response = client.get(reverse("admin:sd_models_llmcosts_changelist"))
        assert response.status_code == 403

    def test_the_call_changelist_renders(self, client, some_calls):
        admin = get_user_model().objects.create_superuser("root", "r@x.invalid", "pw")
        client.force_login(admin)
        assert client.get(reverse("admin:sd_models_llmcall_changelist")).status_code == 200
        assert client.get(reverse("admin:sd_models_llmspan_changelist")).status_code == 200


class TestPrune:
    def test_old_payloads_calls_and_empty_spans_go(self, settings):
        settings.STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS = True
        settings.STRUCKDOWN_LEDGER_PAYLOAD_DAYS = 30
        settings.STRUCKDOWN_LEDGER_CALL_DAYS = 400
        payload = UsagePayload(request={}, response={})
        with llm_span("old"):
            old = ledger.write_record(UsageRecord(kind="chat", model_name="m", payload=payload))
        with llm_span("recent"):
            recent = ledger.write_record(UsageRecord(kind="chat", model_name="m", payload=payload))
        LLMCall.objects.filter(pk=old.pk).update(created_at=timezone.now() - timedelta(days=401))
        LLMCallPayload.objects.filter(call=old).update(
            created_at=timezone.now() - timedelta(days=31)
        )
        LLMSpan.objects.filter(pk=old.span.pk).update(
            started_at=timezone.now() - timedelta(days=401)
        )

        out = StringIO()
        call_command("sd_prune_ledger", "--dry-run", stdout=out)
        assert "Would delete: 1 payloads" in out.getvalue()
        assert LLMCall.objects.count() == 2

        call_command("sd_prune_ledger", stdout=out)
        assert list(LLMCall.objects.values_list("pk", flat=True)) == [recent.pk]
        assert LLMCallPayload.objects.count() == 1
        assert set(LLMSpan.objects.values_list("name", flat=True)) == {"recent"}

    def test_a_zero_window_keeps_everything(self, settings):
        settings.STRUCKDOWN_LEDGER_CALL_DAYS = 0
        settings.STRUCKDOWN_LEDGER_PAYLOAD_DAYS = 0
        call = ledger.write_record(UsageRecord(kind="chat", model_name="m"))
        LLMCall.objects.filter(pk=call.pk).update(created_at=timezone.now() - timedelta(days=5000))
        call_command("sd_prune_ledger", stdout=StringIO())
        assert LLMCall.objects.count() == 1

    def test_the_dry_run_counts_exactly_what_the_real_run_deletes(self, settings):
        settings.STRUCKDOWN_LEDGER_PAYLOAD_DAYS = 0  # payloads go only with their calls
        settings.STRUCKDOWN_LEDGER_CALL_DAYS = 400
        old = timezone.now() - timedelta(days=401)

        def span(name, *call_ages):
            row = LLMSpan.objects.create(name=name, started_at=old)
            for days in call_ages:
                call = LLMCall.objects.create(
                    span=row,
                    kind="chat",
                    model_name="m",
                    created_at=timezone.now() - timedelta(days=days),
                )
                LLMCallPayload.objects.create(call=call, request={}, response={})
            return row

        span("emptied", 401, 402)  # both calls go, so the span does too
        span("empty")
        span("mixed", 401, 10)  # keeps a call, so stays
        young = span("young", 10)
        LLMSpan.objects.filter(pk=young.pk).update(started_at=timezone.now())

        dry = StringIO()
        call_command("sd_prune_ledger", "--dry-run", stdout=dry)
        assert "Would delete: 3 payloads (> 0 d), 3 calls and 2 empty spans (> 400 d)" in (
            dry.getvalue()
        )
        before = (LLMCallPayload.objects.count(), LLMCall.objects.count(), LLMSpan.objects.count())

        real = StringIO()
        call_command("sd_prune_ledger", stdout=real)
        after = (LLMCallPayload.objects.count(), LLMCall.objects.count(), LLMSpan.objects.count())
        assert tuple(b - a for b, a in zip(before, after)) == (3, 3, 2)
        assert real.getvalue().replace("Deleted", "Would delete") == dry.getvalue()
        assert set(LLMSpan.objects.values_list("name", flat=True)) == {"mixed", "young"}
