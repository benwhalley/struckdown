"""Prune the usage ledger: payloads on a short window, calls and spans on a long one.

Usage: python manage.py sd_prune_ledger [--dry-run]

Windows come from settings: STRUCKDOWN_LEDGER_PAYLOAD_DAYS (default 30) and
STRUCKDOWN_LEDGER_CALL_DAYS (default 400, covering a year-on-year comparison).
A window of 0 keeps rows forever. A payload also goes when its call does, and a
span goes once it is past the call window and has no call left after the prune.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand
from django.db.models import Q, QuerySet
from django.utils import timezone

from struckdown.contrib.django.ledger import setting
from struckdown.contrib.django.models import LLMCall, LLMCallPayload, LLMSpan


@dataclass
class PrunePlan:
    payload_days: int
    call_days: int
    payloads: QuerySet
    calls: QuerySet
    spans: QuerySet

    def counts(self) -> tuple[int, int, int]:
        return self.payloads.count(), self.calls.count(), self.spans.count()


def plan(now: datetime) -> PrunePlan:
    """What a prune at ``now`` deletes. The dry run counts it; the real run deletes it.

    None of the three querysets depends on the others' rows being gone, so they
    select the same rows before and after the deletes.
    """
    payload_days = int(setting("PAYLOAD_DAYS", 30))
    call_days = int(setting("CALL_DAYS", 400))

    payload_filter = Q(pk__in=[])
    calls = LLMCall.objects.none()
    spans = LLMSpan.objects.none()
    if payload_days:
        payload_filter |= Q(created_at__lt=now - timedelta(days=payload_days))
    if call_days:
        cutoff = now - timedelta(days=call_days)
        calls = LLMCall.objects.filter(created_at__lt=cutoff)
        # a span whose every call is past the window is empty once they go
        spans = LLMSpan.objects.filter(started_at__lt=cutoff).exclude(
            calls__created_at__gte=cutoff
        )
        payload_filter |= Q(call__created_at__lt=cutoff)
    return PrunePlan(
        payload_days=payload_days,
        call_days=call_days,
        payloads=LLMCallPayload.objects.filter(payload_filter),
        calls=calls,
        spans=spans,
    )


class Command(BaseCommand):
    help = "Delete ledger payloads and calls older than their retention windows."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Count, delete nothing.")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        prune = plan(timezone.now())
        counts = prune.counts()
        if not dry_run:
            prune.payloads.delete()
            prune.calls.delete()
            prune.spans.delete()
        verb = "Would delete" if dry_run else "Deleted"
        self.stdout.write(
            f"{verb}: {counts[0]} payloads (> {prune.payload_days} d), "
            f"{counts[1]} calls and {counts[2]} empty spans (> {prune.call_days} d)"
        )
