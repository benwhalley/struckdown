"""Prune the usage ledger: payloads on a short window, calls and spans on a long one.

Usage: python manage.py sd_prune_ledger [--dry-run]

Windows come from settings: STRUCKDOWN_LEDGER_PAYLOAD_DAYS (default 30) and
STRUCKDOWN_LEDGER_CALL_DAYS (default 400, covering a year-on-year comparison).
A window of 0 keeps rows forever.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from struckdown.contrib.django.ledger import setting
from struckdown.contrib.django.models import LLMCall, LLMCallPayload, LLMSpan


class Command(BaseCommand):
    help = "Delete ledger payloads and calls older than their retention windows."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Count, delete nothing.")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        now = timezone.now()

        payload_days = int(setting("PAYLOAD_DAYS", 30))
        call_days = int(setting("CALL_DAYS", 400))

        payloads = LLMCallPayload.objects.none()
        if payload_days:
            payloads = LLMCallPayload.objects.filter(
                created_at__lt=now - timedelta(days=payload_days)
            )
        calls = LLMCall.objects.none()
        spans = LLMSpan.objects.none()
        if call_days:
            cutoff = now - timedelta(days=call_days)
            calls = LLMCall.objects.filter(created_at__lt=cutoff)
            spans = LLMSpan.objects.filter(started_at__lt=cutoff, calls__isnull=True)

        counts = (payloads.count(), calls.count(), spans.count())
        verb = "Would delete" if dry_run else "Deleted"
        if not dry_run:
            payloads.delete()
            calls.delete()
            # spans whose calls have just gone are now empty too
            spans = LLMSpan.objects.filter(started_at__lt=cutoff, calls__isnull=True) if call_days else spans
            spans.delete()
        self.stdout.write(
            f"{verb}: {counts[0]} payloads (> {payload_days} d), "
            f"{counts[1]} calls and {counts[2]} empty spans (> {call_days} d)"
        )
