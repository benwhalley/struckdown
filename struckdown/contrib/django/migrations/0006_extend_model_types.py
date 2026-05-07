"""Extend AvailableModel.model_type choices with transcription."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sd_models", "0005_remove_inline_credentials"),
    ]

    operations = [
        migrations.AlterField(
            model_name="availablemodel",
            name="model_type",
            field=models.CharField(
                choices=[
                    ("llm", "Language Model"),
                    ("embedding", "Embedding Model"),
                    ("transcription", "Speech-to-Text"),
                ],
                max_length=20,
            ),
        ),
    ]
