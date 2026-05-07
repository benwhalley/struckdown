"""LLM and embedding model configuration.

Three models:
- Credential: API credentials for an LLM provider (shared across models)
- AvailableModel: a model with pricing, linked to a credential
- ModelSet: a collection of models available to users

Provider is derived from the model_name prefix (pydantic-ai convention).
"""

import logging
import uuid
from decimal import Decimal
from typing import Optional
from urllib.parse import urlparse

from django.db import models
from django.utils import timezone
from django_lifecycle import AFTER_SAVE, LifecycleModelMixin, hook

from struckdown.model_spec import PROVIDERS, ModelRegistry, ModelSpec

from .fields import EncryptedCharField

logger = logging.getLogger(__name__)


def generate_short_id():
    """Generate a 22-char hex ID from UUID4."""
    return uuid.uuid4().hex[:22]


class DataResidency(models.TextChoices):
    EU = "eu", "EU"
    US = "us", "US"
    OTHER = "other", "Other"


class PricingSource(models.TextChoices):
    GENAI_PRICES = "genai_prices", "genai-prices"
    OPENROUTER = "openrouter", "OpenRouter API"


class Credential(models.Model):
    """API credentials for an LLM provider.

    Multiple AvailableModels can share a single Credential (e.g. all OpenAI
    models use one API key). ModelSets can have a default Credential as fallback.
    """

    id = models.CharField(
        max_length=50,
        primary_key=True,
        default=generate_short_id,
        editable=False,
    )
    name = models.CharField(
        max_length=100,
        help_text="Human-readable label, e.g. 'OpenAI Production' or 'OpenRouter'",
    )
    api_key = EncryptedCharField(
        max_length=500,
        blank=True,
        help_text="API key for this provider (encrypted at rest).",
    )
    base_url = models.URLField(
        blank=True,
        help_text=(
            "Leave blank for direct provider access (OpenAI, Anthropic, Google, Mistral). "
            "Set for proxies (OpenRouter, LiteLLM), Azure endpoints, or Ollama."
        ),
    )
    pricing_source = models.CharField(
        max_length=20,
        choices=PricingSource.choices,
        default=PricingSource.GENAI_PRICES,
        help_text="Where to look up model pricing for auto-updates.",
    )
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "llm_credential"
        ordering = ["name"]

    def __str__(self):
        masked = self._mask_key()
        url_hint = f" @ {urlparse(self.base_url).hostname}" if self.base_url else ""
        return f"{self.name} ({masked}{url_hint})"

    def _mask_key(self) -> str:
        if not self.api_key:
            return "no key"
        k = self.api_key
        if len(k) <= 10:
            return k[:2] + "***" + k[-2:]
        return k[:4] + "***" + k[-4:]

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


class AvailableModel(LifecycleModelMixin, models.Model):
    """An LLM or embedding model with stored pricing, linked to a credential.

    model_name uses the pydantic-ai ``provider:model`` convention for direct
    provider access (e.g. ``openai:gpt-4o``, ``anthropic:claude-sonnet-4-20250514``).
    Bare names (no prefix) are used for models behind a proxy -- set base_url
    on the credential.

    Credentials are resolved via: model's own credential > model set default
    credential. Models without a resolvable credential will not be usable.
    """

    class ModelType(models.TextChoices):
        LLM = "llm", "Language Model"
        EMBEDDING = "embedding", "Embedding Model"
        TRANSCRIPTION = "transcription", "Speech-to-Text"
        # not yet wired through struckdown -- enable when supported:
        # TTS = "tts", "Text-to-Speech"
        # VISION = "vision", "Vision (Image Understanding)"
        # IMAGE = "image", "Image Generation"
        # RERANKER = "reranker", "Reranker"
        # MODERATION = "moderation", "Moderation"

    id = models.CharField(
        max_length=50,
        primary_key=True,
        default=generate_short_id,
        editable=False,
    )
    model_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text=(
            "provider:model for direct access (e.g. 'openai:gpt-4o'). "
            "Bare name for proxy access (e.g. 'gpt-4.1')."
        ),
    )
    model_type = models.CharField(max_length=20, choices=ModelType.choices)

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    # credential link -- replaces inline api_key/base_url
    credential = models.ForeignKey(
        Credential,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="models",
        help_text="API credential for this model. Falls back to model set default if blank.",
    )

    data_residency = models.CharField(
        max_length=10,
        choices=DataResidency.choices,
        default=DataResidency.US,
        help_text="Where data is processed: EU, US, or Other",
    )

    # stored pricing (replaces live genai-prices lookups)
    input_cost_per_mtok = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Input cost per million tokens (USD).",
    )
    output_cost_per_mtok = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Output cost per million tokens (USD).",
    )
    cost_per_audio_minute = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="USD per minute of audio (for transcription / STT models). Manually set.",
    )
    prices_updated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When pricing was last auto-updated.",
    )
    prices_updated_manually = models.BooleanField(
        default=False,
        help_text="If True, auto-update will not overwrite these prices.",
    )

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "llm_available_model"
        ordering = ["name"]

    def __str__(self):
        cred_hint = ""
        if self.credential_id:
            cred_hint = f" via {self.credential.name}"
        return f"{self.name} ({self.provider_display}{cred_hint})"

    # -- provider properties derived from model_name --

    @property
    def provider(self) -> str:
        """Provider derived from model_name prefix, or 'openai-compatible' for bare names."""
        if ":" in self.model_name:
            return self.model_name.split(":", 1)[0]
        return "openai-compatible"

    @property
    def provider_display(self) -> str:
        """Human-readable provider name."""
        info = PROVIDERS.get(self.provider)
        return info.name if info else self.provider

    @property
    def bare_model_name(self) -> str:
        """Model name without provider prefix (e.g. 'gpt-4o' from 'openai:gpt-4o')."""
        if ":" in self.model_name:
            return self.model_name.split(":", 1)[1]
        return self.model_name

    @property
    def display_name_with_provider(self) -> str:
        """Unambiguous display name including provider source."""
        cred = self.resolve_credential()
        if cred and cred.base_url:
            host = urlparse(cred.base_url).hostname or cred.base_url
            return f"{self.name} via {host}"
        return f"{self.name} ({self.provider_display})"

    # -- credential resolution --

    def resolve_credential(
        self, default_credential: Optional[Credential] = None
    ) -> Optional[Credential]:
        """Return the effective Credential for this model.

        Resolution order:
        1. This model's own credential FK
        2. Explicit default_credential passed by caller (e.g. from a ModelSet)
        3. Default credential from any active ModelSet this model belongs to
        4. None (no credentials available)

        Only active credentials are returned.
        """
        if self.credential_id:
            cred = self.credential
            if cred.is_active:
                return cred
        if default_credential is not None and default_credential.is_active:
            return default_credential
        for ms in self.model_sets.filter(
            is_active=True,
            default_credential__isnull=False,
            default_credential__is_active=True,
        ):
            return ms.default_credential
        return None

    @property
    def api_key(self) -> str:
        """Resolve API key from credential chain. Backwards compatible."""
        cred = self.resolve_credential()
        if cred and cred.api_key:
            return cred.api_key
        return ""

    @property
    def base_url(self) -> str:
        """Resolve base_url from credential chain. Backwards compatible."""
        cred = self.resolve_credential()
        if cred and cred.base_url:
            return cred.base_url
        return ""

    @property
    def has_credentials(self) -> bool:
        """True if this model can resolve an API key."""
        cred = self.resolve_credential()
        return bool(cred and cred.api_key)

    # -- conversion to struckdown types --

    def to_spec(
        self, default_credential: Optional[Credential] = None
    ) -> ModelSpec:
        """Convert to a portable ModelSpec, including stored pricing."""
        cred = self.resolve_credential(default_credential=default_credential)
        return ModelSpec(
            model_name=self.model_name,
            model_type=self.model_type,
            api_key=(cred.api_key if cred else None) or None,
            base_url=(cred.base_url if cred else None) or None,
            data_residency=self.data_residency or None,
            display_name=self.name or None,
            input_cost_per_mtok=(
                float(self.input_cost_per_mtok)
                if self.input_cost_per_mtok is not None
                else None
            ),
            output_cost_per_mtok=(
                float(self.output_cost_per_mtok)
                if self.output_cost_per_mtok is not None
                else None
            ),
        )

    def get_llm_and_credentials(self):
        """Return (LLM, LLMCredentials) tuple ready for struckdown calls.

        Also sets the pricing context var so struckdown uses the stored
        per-mtok rates for cost calculation. Prefer to_spec() for new code.
        """
        from struckdown.audio import set_audio_pricing
        from struckdown.llm import LLM, LLMCredentials, set_model_pricing

        cred = self.resolve_credential()
        set_model_pricing(
            float(self.input_cost_per_mtok) if self.input_cost_per_mtok is not None else None,
            float(self.output_cost_per_mtok) if self.output_cost_per_mtok is not None else None,
        )
        set_audio_pricing(
            float(self.cost_per_audio_minute)
            if self.cost_per_audio_minute is not None
            else None
        )
        return (
            LLM(model_name=self.model_name),
            LLMCredentials(
                api_key=(cred.api_key if cred else None),
                base_url=(cred.base_url if cred else None) or None,
            ),
        )

    # -- pricing --

    @property
    def input_cost_per_token(self) -> Optional[float]:
        """Input cost per token (USD). Reads from stored pricing."""
        if self.input_cost_per_mtok is not None:
            return float(self.input_cost_per_mtok / 1_000_000)
        return None

    @property
    def output_cost_per_token(self) -> Optional[float]:
        """Output cost per token (USD). Reads from stored pricing."""
        if self.output_cost_per_mtok is not None:
            return float(self.output_cost_per_mtok / 1_000_000)
        return None

    @property
    def context_window(self) -> Optional[int]:
        """Context window from genai-prices (cheap lookup, no API call)."""
        m = self._get_genai_model()
        return m.context_window if m else None

    def _get_genai_model(self):
        """Look up model in genai-prices database."""
        try:
            from genai_prices import Usage, calc_price
            from genai_prices.data_snapshot import get_snapshot

            bare = self.bare_model_name
            try:
                result = calc_price(
                    Usage(input_tokens=1_000_000, output_tokens=0), bare
                )
                if result and result.model:
                    return result.model
            except Exception:
                pass

            snap = get_snapshot()
            for provider in snap.providers:
                model = provider.find_model(bare)
                if model:
                    return model
            return None
        except Exception:
            return None

    @hook(AFTER_SAVE)
    def _autofill_prices_on_save(self):
        if self.input_cost_per_mtok is not None or self.output_cost_per_mtok is not None:
            return
        try:
            self.update_prices()
        except Exception:
            logger.exception("update_prices failed for %s", self.model_name)

    def update_prices(self, force: bool = False) -> bool:
        """Look up and store pricing from the credential's pricing source.

        Returns True if prices were updated, False otherwise.
        Skips update if prices_updated_manually is True (unless force=True).
        """
        from struckdown.pricing import get_price_source

        if self.prices_updated_manually and not force:
            return False

        cred = self.resolve_credential()
        source_name = cred.pricing_source if cred else PricingSource.GENAI_PRICES
        source = get_price_source(source_name)

        result = source.lookup(self.model_name, provider=self.provider)
        if result is None:
            return False

        self.input_cost_per_mtok = result.input_cost_per_mtok
        self.output_cost_per_mtok = result.output_cost_per_mtok
        self.prices_updated_at = timezone.now()
        self.save(
            update_fields=[
                "input_cost_per_mtok",
                "output_cost_per_mtok",
                "prices_updated_at",
            ]
        )
        return True


class ModelSet(models.Model):
    """A collection of models available to users.

    One ModelSet is the system default. Models in a set can span providers.
    Convertible to a ModelRegistry for use in pipelines.
    """

    id = models.CharField(
        max_length=50,
        primary_key=True,
        default=generate_short_id,
        editable=False,
    )

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_default = models.BooleanField(
        default=False,
        help_text="The default model set used for new runs and comparisons",
    )

    available_models = models.ManyToManyField(
        AvailableModel,
        related_name="model_sets",
        blank=True,
    )

    default_llm = models.ForeignKey(
        AvailableModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
        limit_choices_to={"model_type": AvailableModel.ModelType.LLM},
    )
    default_embedding_model = models.ForeignKey(
        AvailableModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
        limit_choices_to={"model_type": AvailableModel.ModelType.EMBEDDING},
    )

    default_credential = models.ForeignKey(
        Credential,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="default_for_model_sets",
        help_text="Default credential for models in this set that don't have their own.",
    )

    # optional alias definitions: {"default": "gpt-5-mini", "best": "gpt-5"}
    aliases = models.JSONField(default=dict, blank=True)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "llm_model_set"
        ordering = ["-is_default", "name"]

    def __str__(self):
        default_marker = " (default)" if self.is_default else ""
        return f"{self.name}{default_marker}"

    def save(self, *args, **kwargs):
        if self.is_default:
            ModelSet.objects.filter(is_default=True).exclude(pk=self.pk).update(
                is_default=False
            )
        super().save(*args, **kwargs)

    def get_llms(self):
        """Get all active LLM models in this set."""
        return self.available_models.filter(
            model_type=AvailableModel.ModelType.LLM,
            is_active=True,
        )

    def get_embedding_models(self):
        """Get all embedding models in this set."""
        return self.available_models.filter(
            model_type=AvailableModel.ModelType.EMBEDDING,
            is_active=True,
        )

    @property
    def data_regions(self) -> set:
        """Set of unique data residency values across all active models."""
        return set(
            self.available_models.filter(is_active=True).values_list(
                "data_residency", flat=True
            )
        )

    @property
    def data_residency_summary(self) -> str:
        """Human-readable summary of data residency for this model set."""
        regions = self.data_regions
        labels = dict(DataResidency.choices)
        if not regions:
            return "No models"
        if len(regions) == 1:
            region = next(iter(regions))
            return f"{labels.get(region, region)} only"
        return "Mixed: " + ", ".join(sorted(labels.get(r, r) for r in regions))

    def to_registry(self) -> ModelRegistry:
        """Convert this ModelSet to a ModelRegistry for use in pipelines."""
        specs = {
            m.model_name: m.to_spec(default_credential=self.default_credential)
            for m in self.available_models.filter(is_active=True)
        }
        return ModelRegistry(
            models=specs,
            aliases=self.aliases or {},
            default_llm=(
                self.default_llm.model_name if self.default_llm else None
            ),
            default_embedding=(
                self.default_embedding_model.model_name
                if self.default_embedding_model
                else None
            ),
        )


# -- convenience query functions --


def get_default_model_set(user=None):
    """Get the default ModelSet for the system, or user's preferred set."""
    if user and hasattr(user, "preferred_model_set") and user.preferred_model_set:
        return user.preferred_model_set
    return ModelSet.objects.filter(is_default=True, is_active=True).first()


def get_available_llms(user=None):
    """Get all LLMs from the default model set, or all active LLMs if no set configured."""
    model_set = get_default_model_set(user)
    if model_set:
        return model_set.get_llms()
    return AvailableModel.objects.filter(
        model_type=AvailableModel.ModelType.LLM,
        is_active=True,
    )


def get_available_embedding_models(user=None):
    """Get all embedding models from the default model set."""
    model_set = get_default_model_set(user)
    if model_set:
        return model_set.get_embedding_models()
    return AvailableModel.objects.filter(
        model_type=AvailableModel.ModelType.EMBEDDING,
        is_active=True,
    )


def get_embedding_model_by_name(model_name: str) -> AvailableModel:
    """Get an embedding model by its model_name string."""
    return AvailableModel.objects.get(
        model_name=model_name,
        model_type=AvailableModel.ModelType.EMBEDDING,
        is_active=True,
    )


def get_model_by_id(model_id: str) -> AvailableModel:
    """Get an AvailableModel by its ID or ID prefix."""
    try:
        return AvailableModel.objects.get(id=model_id)
    except AvailableModel.DoesNotExist:
        pass

    if len(model_id) < 22:
        matches = list(AvailableModel.objects.filter(id__startswith=model_id)[:2])
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise AvailableModel.MultipleObjectsReturned(
                f"Multiple models match prefix '{model_id}'"
            )

    raise AvailableModel.DoesNotExist(f"No model found matching '{model_id}'")


def get_model_by_name(model_name: str) -> AvailableModel:
    """Get an AvailableModel by its model_name."""
    return AvailableModel.objects.get(
        model_name=model_name,
        is_active=True,
    )


def get_default_llm_id() -> str:
    """Get the default LLM model ID from the default model set."""
    model_set = get_default_model_set()
    if model_set and model_set.default_llm:
        return str(model_set.default_llm.id)

    llms = get_available_llms()
    first_llm = llms.first()
    if first_llm:
        return str(first_llm.id)

    raise ValueError("No LLM models configured in the system")
