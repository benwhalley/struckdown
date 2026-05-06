"""Idempotent seed command for recommended LLM stubs.

Reads the recommended_models.json fixture and creates any AvailableModel
entries that don't already exist (matched on model_name). Does not overwrite
existing records, so user-set credentials and customisations are preserved.

Run after initial setup: python manage.py sd_seed_models
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from struckdown.contrib.django.models import AvailableModel, ModelSet

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "fixtures" / "recommended_models.json"
)


class Command(BaseCommand):
    help = "Seed recommended LLM stubs from the recommended_models fixture (no API keys)."

    def handle(self, *args, **options):
        with FIXTURE_PATH.open() as f:
            entries = json.load(f)

        created_models = []
        for entry in entries:
            fields = entry["fields"]
            model_name = fields["model_name"]
            obj, created = AvailableModel.objects.get_or_create(
                model_name=model_name,
                defaults={
                    "model_type": fields.get("model_type", "llm"),
                    "name": fields.get("name", model_name),
                    "data_residency": fields.get("data_residency", "us"),
                    "is_active": fields.get("is_active", True),
                },
            )
            status = "created" if created else "exists"
            self.stdout.write(f"  {status}: {obj.name} ({obj.model_name})")
            created_models.append(obj)

        model_set, ms_created = ModelSet.objects.get_or_create(
            is_default=True,
            defaults={"name": "Default", "description": "System default model set"},
        )
        if ms_created:
            model_set.available_models.set(created_models)
            llms = [m for m in created_models if m.model_type == "llm"]
            embeddings = [m for m in created_models if m.model_type == "embedding"]
            if llms:
                model_set.default_llm = llms[0]
            if embeddings:
                model_set.default_embedding_model = embeddings[0]
            model_set.save()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Created default ModelSet with {len(created_models)} models"
                )
            )
        else:
            self.stdout.write(f"Default ModelSet already exists: {model_set.name}")

        self.stdout.write(
            self.style.WARNING(
                "\nNext: configure API keys via admin for each model you want to use."
            )
        )
