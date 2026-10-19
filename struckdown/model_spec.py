"""Canonical model specification types for LLM configuration.

ModelSpec and ModelRegistry provide a portable, framework-agnostic way to
specify models, credentials, and aliases. Downstream projects (soaking, Django
apps) build on these types rather than passing LLM + LLMCredentials separately.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, SecretStr, computed_field


logger = logging.getLogger(__name__)


class ProviderInfo(BaseModel):
    """Metadata about an LLM provider."""

    name: str
    default_residency: str = "other"
    key_env_var: Optional[str] = None
    endpoint_env_var: Optional[str] = None
    docs_url: str = ""


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="OpenAI",
        default_residency="us",
        key_env_var="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/docs",
    ),
    "openai-chat": ProviderInfo(
        name="OpenAI",
        default_residency="us",
        key_env_var="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/docs",
    ),
    "anthropic": ProviderInfo(
        name="Anthropic",
        default_residency="us",
        key_env_var="ANTHROPIC_API_KEY",
        docs_url="https://docs.anthropic.com",
    ),
    "google": ProviderInfo(
        name="Google",
        default_residency="us",
        key_env_var="GOOGLE_API_KEY",
        docs_url="https://ai.google.dev/docs",
    ),
    "google-gla": ProviderInfo(
        name="Google",
        default_residency="us",
        key_env_var="GEMINI_API_KEY",
        docs_url="https://ai.google.dev/docs",
    ),
    "google-vertex": ProviderInfo(
        name="Google Vertex",
        default_residency="us",
        key_env_var="GOOGLE_API_KEY",
        docs_url="https://cloud.google.com/vertex-ai/docs",
    ),
    "mistral": ProviderInfo(
        name="Mistral",
        default_residency="eu",
        key_env_var="MISTRAL_API_KEY",
        docs_url="https://docs.mistral.ai",
    ),
    "azure": ProviderInfo(
        name="Azure OpenAI",
        default_residency="other",
        key_env_var="AZURE_OPENAI_API_KEY",
        endpoint_env_var="AZURE_OPENAI_ENDPOINT",
        docs_url="https://learn.microsoft.com/azure/ai-services/openai/",
    ),
    "ollama": ProviderInfo(
        name="Ollama",
        default_residency="other",
        docs_url="https://ollama.com",
    ),
    "typesafe": ProviderInfo(
        name="TypeSafe (decision models)",
        default_residency="us",
        key_env_var="TYPESAFE_API_KEY",
        docs_url="https://docs.typesafe.ai",
    ),
    "openai-compatible": ProviderInfo(
        name="OpenAI-Compatible",
        default_residency="other",
    ),
}


class ModelSpec(BaseModel):
    """Complete specification for a model endpoint.

    Combines identity (model_name), credentials (api_key, base_url),
    pricing, and metadata (type, residency) into a single portable object.

    When pricing fields are set, struckdown uses them for cost calculation
    instead of looking up prices via pydantic-ai or genai-prices. This
    allows callers to supply known-correct pricing (e.g. from OpenRouter)
    regardless of whether struckdown is used standalone or via Django.
    """

    model_name: str = Field(description="Model identifier in provider:model format")
    model_type: Literal["llm", "embedding", "transcription", "decision"] = "llm"
    alias: Optional[str] = Field(
        default=None, description="Short name templates use in a slot's model= option"
    )
    api_key: Optional[SecretStr] = Field(default=None, repr=False)
    base_url: Optional[str] = None
    data_residency: Optional[str] = None
    display_name: Optional[str] = None

    # pricing (USD per million tokens) -- when set, used for cost calculation
    input_cost_per_mtok: Optional[float] = Field(
        default=None,
        description="Input cost per million tokens (USD). Used for cost calculation if set.",
    )
    output_cost_per_mtok: Optional[float] = Field(
        default=None,
        description="Output cost per million tokens (USD). Used for cost calculation if set.",
    )
    cache_read_cost_per_mtok: Optional[float] = Field(
        default=None,
        description="Price of a cached prompt token read (USD per Mtok). Input rate if unset.",
    )
    cache_write_cost_per_mtok: Optional[float] = Field(
        default=None,
        description="Price of writing a prompt token to cache (USD per Mtok). Input rate if unset.",
    )
    # the caller's own identifier for this model (a database row id, say),
    # carried onto every usage record so a ledger can join back without
    # matching on the name
    model_ref: Optional[str] = None

    @computed_field
    @property
    def provider(self) -> str:
        if ":" in self.model_name:
            return self.model_name.split(":", 1)[0]
        return "openai-compatible"

    @computed_field
    @property
    def bare_name(self) -> str:
        if ":" in self.model_name:
            return self.model_name.split(":", 1)[1]
        return self.model_name

    @property
    def provider_info(self) -> Optional[ProviderInfo]:
        return PROVIDERS.get(self.provider)

    @property
    def provider_display(self) -> str:
        info = self.provider_info
        return info.name if info else self.provider

    def as_llm(self) -> "LLM":
        """Convert to struckdown's internal LLM object."""
        from .llm import LLM

        return LLM(
            model_name=self.model_name,
            model_type="decision" if self.model_type == "decision" else "llm",
            model_ref=str(self.model_ref) if self.model_ref is not None else "",
        )

    def as_credentials(self) -> "LLMCredentials":
        """Convert to struckdown's internal LLMCredentials object."""
        from .llm import LLMCredentials

        return LLMCredentials(
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
        )


class ModelRegistry(BaseModel):
    """Collection of models with alias resolution.

    Replaces the pattern of passing separate model_name + models dict +
    llm_credentials. Downstream code resolves aliases or model names
    through a single interface.
    """

    models: Dict[str, ModelSpec] = Field(default_factory=dict)
    aliases: Dict[str, str] = Field(default_factory=dict)
    default_llm: Optional[str] = None
    default_embedding: Optional[str] = None
    default_decision: Optional[str] = None

    @classmethod
    def from_specs(cls, specs: List[ModelSpec], **kwargs) -> "ModelRegistry":
        """Build a registry from a list of specs; each spec's `alias` is registered."""
        registry = cls(**kwargs)
        for spec in specs:
            registry.register(spec)
        return registry

    def register(self, spec: ModelSpec) -> None:
        """Add a model to the registry."""
        self.models[spec.model_name] = spec
        if spec.alias:
            self.aliases[spec.alias] = spec.model_name

    def match(self, name: str) -> Tuple[ModelSpec, bool]:
        """Find a registered model by alias or name; return (spec, exact).

        Exact: an alias, model_name or bare name. Otherwise a case-insensitive substring
        match against those and display_name, which must be unique. Raises ValueError if
        nothing (or more than one model) matches -- unlike resolve(), never invents a spec.
        """
        target = self.aliases.get(name, name)
        exact = self.models.get(target) or next(
            (s for s in self.models.values() if s.bare_name == target), None
        )
        if exact:
            return exact, True

        def names(spec):
            aliases = [a for a, m in self.aliases.items() if m == spec.model_name]
            return [spec.model_name, spec.display_name or "", *aliases]

        needle = name.lower()
        hits = [s for s in self.models.values() if any(needle in n.lower() for n in names(s))]
        available = sorted([*self.aliases, *self.models])
        if not hits:
            raise ValueError(f"No model matches '{name}'. Available: {available}")
        if len(hits) > 1:
            raise ValueError(
                f"'{name}' matches several models: {[h.model_name for h in hits]}"
            )
        return hits[0], False

    def resolve(self, name_or_alias: Optional[str] = None) -> ModelSpec:
        """Resolve a name or alias to a ModelSpec.

        Resolution order:
        1. None -> default_llm
        2. Exact alias match -> follow to model_name
        3. Exact model_name match in registry
        4. Bare name match (strip provider prefix from registered models)
        5. Return a minimal spec with just the name (caller must supply credentials)
        """
        if name_or_alias is None:
            name_or_alias = self.default_llm
        if name_or_alias is None:
            raise ValueError("No model specified and no default_llm set")

        # alias lookup
        resolved_name = self.aliases.get(name_or_alias, name_or_alias)

        # exact match
        if resolved_name in self.models:
            return self.models[resolved_name]

        # bare name match -- the resolved_name might be a bare name that matches
        # a registered model with a provider prefix
        for spec in self.models.values():
            if spec.bare_name == resolved_name:
                return spec

        # not registered -- return a minimal spec (credentials from env)
        return ModelSpec(model_name=resolved_name)

    def resolve_embedding(self, name: Optional[str] = None) -> ModelSpec:
        """Resolve an embedding model name or alias."""
        name = name or self.default_embedding
        if name is None:
            raise ValueError("No embedding model specified and no default_embedding set")
        return self.resolve(name)

    def resolve_decision(self, name: Optional[str] = None) -> ModelSpec:
        """The decision model to use: `name`, else default_decision, else the only one."""
        name = name or self.default_decision
        if name:
            return self.match(name)[0]
        decisions = self.decisions()
        if len(decisions) != 1:
            raise ValueError(
                f"No decision model available: set default_decision or pass exactly one "
                f"decision model (found {[d.model_name for d in decisions]})"
            )
        return decisions[0]

    def decisions(self) -> List[ModelSpec]:
        """All registered decision-model specs."""
        return [s for s in self.models.values() if s.model_type == "decision"]

    def llms(self) -> List[ModelSpec]:
        """All registered LLM specs."""
        return [s for s in self.models.values() if s.model_type == "llm"]

    def embeddings(self) -> List[ModelSpec]:
        """All registered embedding specs."""
        return [s for s in self.models.values() if s.model_type == "embedding"]

    @classmethod
    def from_env(cls) -> "ModelRegistry":
        """Build a minimal registry from environment variables.

        Uses DEFAULT_LLM / LLM_API_KEY / LLM_API_BASE.
        """
        from .llm import env_config

        model_name = env_config("DEFAULT_LLM", "gpt-4.1-mini")
        api_key = env_config("LLM_API_KEY", None)
        base_url = env_config("LLM_API_BASE", None)
        spec = ModelSpec(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        return cls(
            models={model_name: spec},
            default_llm=model_name,
        )
