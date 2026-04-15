"""Re-export encrypted fields from django-fernet-encrypted-fields.

Provides EncryptedCharField and EncryptedTextField for model use.
Requires SALT_KEY in Django settings (SECRET_KEY is used automatically).
"""

from encrypted_fields.fields import EncryptedCharField, EncryptedTextField

__all__ = ["EncryptedCharField", "EncryptedTextField"]
