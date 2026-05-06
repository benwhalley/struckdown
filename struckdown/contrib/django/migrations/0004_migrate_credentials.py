"""Data migration: extract inline api_key/base_url from AvailableModel into
Credential rows.

Groups models by (api_key, base_url) pair and creates one Credential per
unique pair. Uses raw SQL to copy encrypted api_key values as-is (same
EncryptedCharField, same encryption key).
"""

from django.db import migrations

from struckdown.contrib.django.models import generate_short_id


def migrate_credentials_forward(apps, schema_editor):
    """Create Credential rows from existing AvailableModel inline credentials."""
    db_alias = schema_editor.connection.alias
    AvailableModel = apps.get_model("sd_models", "AvailableModel")
    Credential = apps.get_model("sd_models", "Credential")

    # group models by (api_key, base_url) -- api_key is the raw encrypted value
    from collections import defaultdict

    groups = defaultdict(list)
    for am in AvailableModel.objects.using(db_alias).exclude(api_key=""):
        key = (am.api_key, am.base_url)
        groups[key].append(am)

    for (api_key, base_url), model_list in groups.items():
        # derive a name from the first model's provider
        first = model_list[0]
        model_name = first.model_name
        if ":" in model_name:
            provider = model_name.split(":", 1)[0]
        else:
            provider = "proxy"

        name = f"{provider} (migrated)"
        if base_url:
            from urllib.parse import urlparse

            host = urlparse(base_url).hostname or base_url
            name = f"{provider} via {host} (migrated)"

        cred = Credential.objects.using(db_alias).create(
            id=generate_short_id(),
            name=name,
            api_key=api_key,
            base_url=base_url,
            description="Auto-created during credential migration.",
        )

        for am in model_list:
            am.credential = cred
            am.save(update_fields=["credential"])


def migrate_credentials_backward(apps, schema_editor):
    """Copy credentials back to inline fields on AvailableModel."""
    db_alias = schema_editor.connection.alias
    AvailableModel = apps.get_model("sd_models", "AvailableModel")

    for am in (
        AvailableModel.objects.using(db_alias)
        .select_related("credential")
        .exclude(credential=None)
    ):
        am.api_key = am.credential.api_key or ""
        am.base_url = am.credential.base_url or ""
        am.save(update_fields=["api_key", "base_url"])


class Migration(migrations.Migration):

    dependencies = [
        ("sd_models", "0003_credential_and_pricing"),
    ]

    operations = [
        migrations.RunPython(
            migrate_credentials_forward,
            migrate_credentials_backward,
        ),
    ]
