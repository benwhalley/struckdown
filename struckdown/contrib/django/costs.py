"""Aggregations behind the costs page. Plain dicts, no HTML.

Every figure here is an estimate from token counts and list prices; the
provider's invoice is the truth. Calls the ledger could not price are counted
and shown, never folded into a total as zero.
"""

from __future__ import annotations

import statistics
from datetime import timedelta
from decimal import Decimal
from typing import Optional

from django.db.models import Case, Count, F, IntegerField, Q, Sum, Value, When
from django.db.models.functions import Coalesce, TruncMonth
from django.utils import timezone

WINDOWS = ((1, "24 h"), (7, "7 d"), (30, "30 d"), (365, "1 y"), (0, "All"))

UNATTRIBUTED = "unattributed"

_unpriced = Q(input_cost__isnull=True, ok=True, cache_hit=False)


def _window(days: int):
    from .models import LLMCall

    calls = LLMCall.objects.all()
    if days:
        calls = calls.filter(created_at__gte=timezone.now() - timedelta(days=days))
    return calls


def _totals():
    return dict(
        calls=Count("id"),
        users=Count("span__user", distinct=True),
        input_tokens=Coalesce(Sum("input_tokens"), 0),
        cache_read_tokens=Coalesce(Sum("cache_read_tokens"), 0),
        output_tokens=Coalesce(Sum("output_tokens"), 0),
        cost_in=Sum("input_cost"),
        cost_out=Sum("output_cost"),
        cost_total=Sum("total_cost"),
        unpriced=Sum(Case(When(_unpriced, then=1), default=0, output_field=IntegerField())),
        cache_hits=Sum(Case(When(cache_hit=True, then=1), default=0, output_field=IntegerField())),
        failed=Sum(Case(When(ok=False, then=1), default=0, output_field=IntegerField())),
    )


def _finish(row: dict) -> dict:
    """Derived columns a template should not compute."""
    calls = row.get("calls") or 0
    priced_calls = calls - (row.get("unpriced") or 0) - (row.get("cache_hits") or 0)
    total = row.get("cost_total")
    row["per_call"] = (total / priced_calls) if total is not None and priced_calls else None
    input_tokens = row.get("input_tokens") or 0
    row["cache_share"] = (
        round(100 * (row.get("cache_read_tokens") or 0) / input_tokens) if input_tokens else 0
    )
    users = row.get("users") or 0
    row["per_user"] = (total / users) if total is not None and users else None
    return row


def headline(days: int) -> dict:
    return _finish(_window(days).aggregate(**_totals()))


def by_feature(days: int) -> list[dict]:
    rows = (
        _window(days)
        .annotate(feature=Case(When(root_name="", then=Value(UNATTRIBUTED)), default=F("root_name")))
        .values("feature")
        .annotate(**_totals())
        .order_by("-cost_total", "-calls")
    )
    return [_finish(dict(r)) for r in rows]


def by_model(days: int) -> list[dict]:
    rows = (
        _window(days)
        .values("model_name", "provider", "kind")
        .annotate(**_totals())
        .order_by("-cost_total", "-calls")
    )
    return [_finish(dict(r)) for r in rows]


def by_month(months: int = 12) -> list[dict]:
    """The last ``months`` calendar months, regardless of the window."""
    since = (timezone.now().replace(day=1) - timedelta(days=31 * (months - 1))).replace(day=1)
    rows = (
        _window(0)
        .filter(created_at__gte=since)
        .annotate(month=TruncMonth("created_at"))
        .values("month")
        .annotate(**_totals())
        .order_by("month")
    )
    return [_finish(dict(r)) for r in rows]


def per_user(days: int) -> dict:
    """Counts and averages only. Nobody is named here.

    A named lookup belongs in the permissioned changelists.
    """
    rows = (
        _window(days)
        .exclude(span__user__isnull=True)
        .values("span__user")
        .annotate(total=Sum("total_cost"), calls=Count("id"))
    )
    totals = [float(r["total"]) for r in rows if r["total"] is not None]
    return {
        "users": len(rows),
        "mean": statistics.fmean(totals) if totals else None,
        "median": statistics.median(totals) if totals else None,
        "max": max(totals) if totals else None,
    }


def dashboard(days: int) -> dict:
    return {
        "headline": headline(days),
        "by_feature": by_feature(days),
        "by_model": by_model(days),
        "by_month": by_month(),
        "per_user": per_user(days),
    }


def as_money(value: Optional[Decimal | float]) -> str:
    """For tests and shells; the template has its own filter."""
    if value is None:
        return "unknown"
    return f"${float(value):,.4f}"
