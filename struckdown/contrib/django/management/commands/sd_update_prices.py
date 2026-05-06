"""Bulk-refresh pricing for all active AvailableModels.

Skips models with prices_updated_manually unless --force is passed.
Usage: python manage.py sd_update_prices [--force] [--dry-run]
"""

from django.core.management.base import BaseCommand

from struckdown.contrib.django.models import AvailableModel


class Command(BaseCommand):
    help = "Update stored pricing for all active models from their credential's pricing source."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Update even manually-set prices.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be updated without making changes.",
        )

    def handle(self, *args, **options):
        force = options["force"]
        dry_run = options["dry_run"]

        models = AvailableModel.objects.filter(is_active=True)
        if not force:
            models = models.filter(prices_updated_manually=False)

        total = models.count()
        updated = 0
        failed = 0

        for model in models.select_related("credential"):
            cred = model.resolve_credential()
            source = cred.pricing_source if cred else "genai_prices"

            if dry_run:
                self.stdout.write(
                    f"  Would update: {model.name} ({model.model_name}) "
                    f"via {source}"
                )
                continue

            if model.update_prices(force=force):
                updated += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  Updated: {model.name} "
                        f"(in: {model.input_cost_per_mtok}, out: {model.output_cost_per_mtok})"
                    )
                )
            else:
                failed += 1
                self.stdout.write(
                    self.style.WARNING(
                        f"  No pricing found: {model.name} ({model.model_name})"
                    )
                )

        if dry_run:
            self.stdout.write(f"\nDry run: {total} models would be checked.")
        else:
            self.stdout.write(
                f"\nDone: {updated} updated, {failed} not found, "
                f"{total} total checked."
            )
