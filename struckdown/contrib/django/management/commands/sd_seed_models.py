"""Idempotent seed command for common LLM and embedding model stubs.

Creates AvailableModel entries (without API keys) and a default ModelSet.
Run after initial setup: python manage.py sd_seed_models
"""

from django.core.management.base import BaseCommand

from struckdown.contrib.django.models import AvailableModel, ModelSet

DEFAULT_MODELS = [
    {
        "model_name": "openai:gpt-4.1-mini",
        "model_type": "llm",
        "name": "GPT-4.1 Mini",
        "data_residency": "us",
    },
    {
        "model_name": "openai:gpt-4.1",
        "model_type": "llm",
        "name": "GPT-4.1",
        "data_residency": "us",
    },
    {
        "model_name": "openai:gpt-5-mini",
        "model_type": "llm",
        "name": "GPT-5 Mini",
        "data_residency": "us",
    },
    {
        "model_name": "anthropic:claude-sonnet-4-20250514",
        "model_type": "llm",
        "name": "Claude Sonnet 4",
        "data_residency": "us",
    },
    {
        "model_name": "anthropic:claude-haiku-4-5-20251001",
        "model_type": "llm",
        "name": "Claude Haiku 4.5",
        "data_residency": "us",
    },
    {
        "model_name": "google-gla:gemini-2.5-flash",
        "model_type": "llm",
        "name": "Gemini 2.5 Flash",
        "data_residency": "us",
    },
    {
        "model_name": "mistral:mistral-small-latest",
        "model_type": "llm",
        "name": "Mistral Small",
        "data_residency": "eu",
    },
    {
        "model_name": "openai:text-embedding-3-large",
        "model_type": "embedding",
        "name": "Text Embedding 3 Large",
        "data_residency": "us",
    },
    {
        "model_name": "openai:text-embedding-3-small",
        "model_type": "embedding",
        "name": "Text Embedding 3 Small",
        "data_residency": "us",
    },
]


class Command(BaseCommand):
    help = "Seed common LLM and embedding model stubs (no API keys)"

    def handle(self, *args, **options):
        created_models = []
        for entry in DEFAULT_MODELS:
            obj, created = AvailableModel.objects.get_or_create(
                model_name=entry["model_name"],
                defaults={
                    "model_type": entry["model_type"],
                    "name": entry["name"],
                    "data_residency": entry.get("data_residency", "us"),
                },
            )
            status = "created" if created else "exists"
            self.stdout.write(f"  {status}: {obj.name} ({obj.model_name})")
            created_models.append(obj)

        # ensure a default ModelSet exists
        model_set, ms_created = ModelSet.objects.get_or_create(
            is_default=True,
            defaults={"name": "Default", "description": "System default model set"},
        )
        if ms_created:
            model_set.available_models.set(created_models)
            # set defaults
            llms = [m for m in created_models if m.model_type == "llm"]
            embeddings = [m for m in created_models if m.model_type == "embedding"]
            if llms:
                model_set.default_llm = llms[0]
            if embeddings:
                model_set.default_embedding_model = embeddings[0]
            model_set.save()
            self.stdout.write(self.style.SUCCESS(f"Created default ModelSet with {len(created_models)} models"))
        else:
            self.stdout.write(f"Default ModelSet already exists: {model_set.name}")

        self.stdout.write(
            self.style.WARNING(
                "\nNext: configure API keys via admin for each model you want to use."
            )
        )
