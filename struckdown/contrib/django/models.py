"""LLM and embedding model configuration.

Two models:
- AvailableModel: a model with its API credentials, ready to use
- ModelSet: a collection of models available to users

Provider is derived from the model_name prefix (pydantic-ai convention).
"""

import uuid

from django.db import models

from struckdown.model_spec import PROVIDERS, ModelRegistry, ModelSpec

from .fields import EncryptedCharField


def generate_short_id():
    """Generate a 22-char hex ID from UUID4."""
    return uuid.uuid4().hex[:22]


class DataResidency(models.TextChoices):
    EU = "eu", "EU"
    US = "us", "US"
    OTHER = "other", "Other"


class AvailableModel(models.Model):
    """An LLM or embedding model, ready to use.

    model_name uses the pydantic-ai ``provider:model`` convention for direct
    provider access (e.g. ``openai:gpt-4o``, ``anthropic:claude-sonnet-4-20250514``).
    Bare names (no prefix) are used for models behind a proxy -- set base_url.

    Each model stores its own API key and base_url. No separate credential
    model -- everything needed to call the model is right here.
    """

    class ModelType(models.TextChoices):
        LLM = "llm", "Language Model"
        EMBEDDING = "embedding", "Embedding Model"

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

    # credentials -- everything needed to call this model
    api_key = EncryptedCharField(
        max_length=500,
        blank=True,
        help_text="API key for this model's provider (encrypted at rest).",
    )
    base_url = models.URLField(
        blank=True,
        help_text=(
            "Leave blank for direct provider access (OpenAI, Anthropic, Google, Mistral). "
            "Set only for proxies (LiteLLM), Azure endpoints, or Ollama."
        ),
    )

    data_residency = models.CharField(
        max_length=10,
        choices=DataResidency.choices,
        default=DataResidency.US,
        help_text="Where data is processed: EU, US, or Other",
    )

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "llm_available_model"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.provider_display})"

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
    def has_credentials(self) -> bool:
        return bool(self.api_key)

    # -- conversion to struckdown types --

    def to_spec(self) -> ModelSpec:
        """Convert to a portable ModelSpec."""
        return ModelSpec(
            model_name=self.model_name,
            model_type=self.model_type,
            api_key=self.api_key or None,
            base_url=self.base_url or None,
            data_residency=self.data_residency or None,
            display_name=self.name or None,
        )

    def get_llm_and_credentials(self):
        """Return (LLM, LLMCredentials) tuple ready for struckdown calls.

        Convenience method -- prefer to_spec() for new code.
        """
        from struckdown.llm import LLM, LLMCredentials

        return (
            LLM(model_name=self.model_name),
            LLMCredentials(api_key=self.api_key, base_url=self.base_url or None),
        )

    # -- optional pricing properties (require genai-prices) --

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

    @property
    def context_window(self):
        m = self._get_genai_model()
        return m.context_window if m else None

    @property
    def input_cost_per_token(self):
        m = self._get_genai_model()
        if m and m.prices and m.prices.input_mtok is not None:
            from decimal import Decimal

            return float(Decimal(str(m.prices.input_mtok)) / 1_000_000)
        return None

    @property
    def output_cost_per_token(self):
        m = self._get_genai_model()
        if m and m.prices and m.prices.output_mtok is not None:
            from decimal import Decimal

            return float(Decimal(str(m.prices.output_mtok)) / 1_000_000)
        return None


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
            m.model_name: m.to_spec()
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
