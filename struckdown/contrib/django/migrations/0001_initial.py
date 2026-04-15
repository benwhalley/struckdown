"""Initial migration for sd_models.

For new installations: creates both tables from scratch.
For soakresearch (migrating from llm_config): run with --fake-initial
since the tables already exist under the same names.
"""

import django.db.models.deletion
from django.db import migrations, models

import struckdown.contrib.django.models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AvailableModel",
            fields=[
                (
                    "id",
                    models.CharField(
                        default=struckdown.contrib.django.models.generate_short_id,
                        editable=False,
                        max_length=50,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "model_name",
                    models.CharField(
                        db_index=True,
                        help_text=(
                            "provider:model for direct access (e.g. 'openai:gpt-4o'). "
                            "Bare name for proxy access (e.g. 'gpt-4.1')."
                        ),
                        max_length=100,
                    ),
                ),
                (
                    "model_type",
                    models.CharField(
                        choices=[("llm", "Language Model"), ("embedding", "Embedding Model")],
                        max_length=20,
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField(blank=True)),
                (
                    "api_key",
                    models.TextField(
                        blank=True,
                        help_text="API key for this model's provider (encrypted at rest).",
                    ),
                ),
                (
                    "base_url",
                    models.URLField(
                        blank=True,
                        help_text=(
                            "Leave blank for direct provider access (OpenAI, Anthropic, Google, Mistral). "
                            "Set only for proxies (LiteLLM), Azure endpoints, or Ollama."
                        ),
                    ),
                ),
                (
                    "data_residency",
                    models.CharField(
                        choices=[("eu", "EU"), ("us", "US"), ("other", "Other")],
                        default="us",
                        help_text="Where data is processed: EU, US, or Other",
                        max_length=10,
                    ),
                ),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "db_table": "llm_available_model",
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="ModelSet",
            fields=[
                (
                    "id",
                    models.CharField(
                        default=struckdown.contrib.django.models.generate_short_id,
                        editable=False,
                        max_length=50,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField(blank=True)),
                (
                    "is_default",
                    models.BooleanField(
                        default=False,
                        help_text="The default model set used for new runs and comparisons",
                    ),
                ),
                (
                    "available_models",
                    models.ManyToManyField(
                        blank=True,
                        related_name="model_sets",
                        to="sd_models.availablemodel",
                    ),
                ),
                (
                    "default_llm",
                    models.ForeignKey(
                        blank=True,
                        limit_choices_to={"model_type": "llm"},
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="+",
                        to="sd_models.availablemodel",
                    ),
                ),
                (
                    "default_embedding_model",
                    models.ForeignKey(
                        blank=True,
                        limit_choices_to={"model_type": "embedding"},
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="+",
                        to="sd_models.availablemodel",
                    ),
                ),
                ("aliases", models.JSONField(blank=True, default=dict)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "db_table": "llm_model_set",
                "ordering": ["-is_default", "name"],
            },
        ),
    ]
