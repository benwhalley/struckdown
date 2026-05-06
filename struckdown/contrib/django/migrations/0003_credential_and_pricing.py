"""Add Credential model, credential FK on AvailableModel, pricing fields,
and default_credential on ModelSet.

Non-destructive: all new fields are nullable or have defaults.
"""

import django.db.models.deletion
import encrypted_fields.fields
from django.db import migrations, models

import struckdown.contrib.django.models


class Migration(migrations.Migration):

    dependencies = [
        ("sd_models", "0002_alter_availablemodel_api_key"),
    ]

    operations = [
        # -- Create Credential model --
        migrations.CreateModel(
            name="Credential",
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
                    "name",
                    models.CharField(
                        help_text="Human-readable label, e.g. 'OpenAI Production' or 'OpenRouter'",
                        max_length=100,
                    ),
                ),
                (
                    "api_key",
                    encrypted_fields.fields.EncryptedCharField(
                        blank=True,
                        help_text="API key for this provider (encrypted at rest).",
                        max_length=500,
                    ),
                ),
                (
                    "base_url",
                    models.URLField(
                        blank=True,
                        help_text=(
                            "Leave blank for direct provider access (OpenAI, Anthropic, Google, Mistral). "
                            "Set for proxies (OpenRouter, LiteLLM), Azure endpoints, or Ollama."
                        ),
                    ),
                ),
                (
                    "pricing_source",
                    models.CharField(
                        choices=[
                            ("genai_prices", "genai-prices"),
                            ("openrouter", "OpenRouter API"),
                        ],
                        default="genai_prices",
                        help_text="Where to look up model pricing for auto-updates.",
                        max_length=20,
                    ),
                ),
                ("description", models.TextField(blank=True)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "db_table": "llm_credential",
                "ordering": ["name"],
            },
        ),
        # -- Add credential FK to AvailableModel --
        migrations.AddField(
            model_name="availablemodel",
            name="credential",
            field=models.ForeignKey(
                blank=True,
                help_text="API credential for this model. Falls back to model set default if blank.",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="models",
                to="sd_models.credential",
            ),
        ),
        # -- Add pricing fields to AvailableModel --
        migrations.AddField(
            model_name="availablemodel",
            name="input_cost_per_mtok",
            field=models.DecimalField(
                blank=True,
                decimal_places=6,
                help_text="Input cost per million tokens (USD).",
                max_digits=12,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="availablemodel",
            name="output_cost_per_mtok",
            field=models.DecimalField(
                blank=True,
                decimal_places=6,
                help_text="Output cost per million tokens (USD).",
                max_digits=12,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="availablemodel",
            name="prices_updated_at",
            field=models.DateTimeField(
                blank=True,
                help_text="When pricing was last auto-updated.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="availablemodel",
            name="prices_updated_manually",
            field=models.BooleanField(
                default=False,
                help_text="If True, auto-update will not overwrite these prices.",
            ),
        ),
        # -- Add default_credential to ModelSet --
        migrations.AddField(
            model_name="modelset",
            name="default_credential",
            field=models.ForeignKey(
                blank=True,
                help_text="Default credential for models in this set that don't have their own.",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="default_for_model_sets",
                to="sd_models.credential",
            ),
        ),
    ]
