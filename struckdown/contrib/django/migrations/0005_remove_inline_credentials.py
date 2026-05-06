"""Remove api_key and base_url columns from AvailableModel.

These now live on the Credential model. Backwards-compatible properties
on AvailableModel resolve through the credential FK.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("sd_models", "0004_migrate_credentials"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="availablemodel",
            name="api_key",
        ),
        migrations.RemoveField(
            model_name="availablemodel",
            name="base_url",
        ),
    ]
