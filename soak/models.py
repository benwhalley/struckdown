"""Data models for qualitative analysis pipelines."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from chatter import LLM, LLMCredentials, chatter

if TYPE_CHECKING:
    from .dag import QualitativeAnalysisPipeline
from .document_utils import extract_text, get_scrubber, unpack_zip_to_temp_paths_if_needed
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, ConfigDict, Field
import logging

logger = logging.getLogger(__name__)


class Code(BaseModel):
    # slug: str = Field(..., min_length=8, description="Unique identifier for the code")
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(..., min_length=5, description="A description of the code.")
    quotes: List[str] = Field(
        ...,
        min_length=0,
        description="Example quotes from the text which illustrate the code. Choose the best examples.",
    )


class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    def to_markdown(self):
        return "\n\n".join([f"- {i.name}: {i.description}\n{i.quotes}" for i in self.codes])


class Theme(BaseModel):
    name: str = Field(..., min_length=10)
    description: str = Field(..., min_length=10)
    # refer to codes by slug/identifier
    code_names: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of the codes that are part of this theme. Identify them accurately by name",
    )


class Themes(BaseModel):
    themes: List[Theme] = Field(..., min_length=1)

    def to_markdown(self):
        return "\n- ".join([i.name for i in self.themes])


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualitativeAnalysis(BaseModel):
    codes: Optional[List[Code]] = None
    themes: Optional[List[Theme]] = None
    narrative: Optional[str] = None

    details: Dict[str, Any] = Field(default_factory=dict)
    config: Optional["DAGConfig"] = None
    pipeline: Optional[str] = None

    def name(self):
        return self.sha256()[:8]

    def sha256(self):
        return hashlib.sha256(json.dumps(self.model_dump()).encode()).hexdigest()[:8]

    def __str__(self):
        return (
            f"Themes: {self.themes}\nCodes: {self.codes}\nDocuments:"
            f" {len(self.config.get('documents', []))} docs"
        )

    def to_html(self, template_path: Optional[str] = None) -> str:
        """Render the analysis as HTML using Jinja2 template from file.

        Args:
            template_path: Path to the HTML template file. If None, uses default template.

        Returns:
            Rendered HTML string.
        """
        if template_path is None:
            # Use default template in soak/templates directory
            template_dir = Path(__file__).parent / "templates"
            template_name = "qualitative_analysis.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)

        # Render template with data
        return template.render(analysis=self)


class QualitativeAnalysisComparison(BaseModel):
    results: List[QualitativeAnalysis]
    combinations: Dict[str, Tuple[QualitativeAnalysis, QualitativeAnalysis]]
    statistics: Dict[str, Dict]
    comparison_plots: Dict[str, Dict[str, Any]]  # eg. heatmaps.xxxyyy = List
    additional_plots: Dict[str, Any]
    config: dict

    def by_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a nested dict keyed by comparison key, with inner dict containing:
            - 'a', 'b' (the QualitativeAnalysis objects)
            - 'stats'
            - 'plots': { plot_type: plot_object, ... }
        """
        out = {}
        for key, (a, b) in self.combinations.items():
            out[key] = {"a": a, "b": b, "stats": self.statistics.get(key), "plots": {}}

        for plot_type in self.comparison_plots.keys():
            for k in out.keys():
                out[k]["plots"][plot_type] = self.comparison_plots[plot_type][k]

        return out


class DAGConfig(BaseModel):
    document_paths: Optional[List[str]] = []
    documents: List[str] = []
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: Optional[LLMCredentials] = None
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: Optional[str] = Field(default="42", exclude=True)

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    def load_documents(self) -> List[str]:
        if hasattr(self, "documents") and self.documents:
            logger.info("Using cached documents")
            return self.documents

        with unpack_zip_to_temp_paths_if_needed(self.document_paths) as dp_:
            self.documents = [extract_text(i) for i in dp_]

        if self.scrub_pii:
            logger.info("Scrubbing PII")
            if self.scrubber_salt == 42:
                logger.warning("Scrubber salt is default, consider setting to a random value")

            scrubber = get_scrubber(model=self.scrubber_model, salt=self.scrubber_salt)
            self.documents = [scrubber.clean(i) for i in self.documents]

        return self.documents
