"""Store a hash of each token, not the token.

Existing tokens are hashed where they are, so values already handed out keep
working. Reversing cannot recover them: it gives every token a new value."""

import hashlib

import shortuuid
from django.db import migrations, models


def hash_existing(apps, schema_editor):
    """The same hash as ``models.hash_token``, kept here so later changes to the
    model's code cannot change what this migration does."""
    RouteToken = apps.get_model("sd_routes", "RouteToken")
    for token in RouteToken.objects.all():
        token.token_hash = hashlib.sha256(token.token.encode()).hexdigest()
        token.prefix = token.token[:4]
        token.save(update_fields=["token_hash", "prefix"])


def fresh_tokens(apps, schema_editor):
    """Reversing: the old values cannot be recovered from their hashes, so every
    token gets a new one, stored in plain text again. Callers need the new values."""
    RouteToken = apps.get_model("sd_routes", "RouteToken")
    for token in RouteToken.objects.all():
        token.token = shortuuid.uuid()
        token.save(update_fields=["token"])


class Migration(migrations.Migration):

    dependencies = [
        ("sd_routes", "0003_short_tokens"),
    ]

    operations = [
        migrations.AddField(
            model_name="routetoken",
            name="token_hash",
            field=models.CharField(editable=False, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name="routetoken",
            name="prefix",
            field=models.CharField(
                default="",
                editable=False,
                help_text="The token's first characters, to tell it apart.",
                max_length=4,
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="routetoken",
            name="token",
            field=models.CharField(editable=False, max_length=40, null=True),
        ),
        migrations.RunPython(hash_existing, fresh_tokens),
        migrations.AlterField(
            model_name="routetoken",
            name="token_hash",
            field=models.CharField(editable=False, max_length=64, unique=True),
        ),
        migrations.RemoveField(
            model_name="routetoken",
            name="token",
        ),
    ]
