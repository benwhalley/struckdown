"""Data models for qualitative analysis pipelines."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, ConfigDict, Field


class Code(BaseModel):
    # slug: str = Field(..., min_length=8, description="Unique identifier for the code")
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=5)
    quotes: List[str] = Field(..., min_length=0)

    def __str__(self):
        return f"{self.name}: {self.description}. Example quotes:{';'.join(self.quotes)}."

    @classmethod
    def mock_(self):
        return Code(
            slug=str(uuid.uuid4()),
            name=str(uuid.uuid4()),
            description=str(uuid.uuid4()),
            quotes=["blah"],
        )


class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    def __str__(self):
        return "\n\n".join(map(str, self.codes))

    def to_markdown(self):
        return "\n\n".join([f"- {i.name}: {i.description}\n{i.quotes}" for i in self.codes])

    @classmethod
    def mock_(self):
        return CodeList(
            codes=[
                Code(
                    slug=str(uuid.uuid4()),
                    name=str(uuid.uuid4()),
                    description=str(uuid.uuid4()),
                    quotes=["blah"],
                )
            ]
        )


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

    @classmethod
    def mock_(self):
        return Themes(
            themes=[
                Theme(
                    name=str(uuid.uuid4()),
                    description=str(uuid.uuid4()),
                    codes=[Code.mock_()],
                )
            ]
        )


class ResearchFinding(BaseModel):
    description: str = Field(..., min_length=50, description="An inference made from some data.")


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""

    seed: int = 2025
    model_name: str = "gpt-4o-mini"
    pipeline_name: str = "rta"
    temperature: float = 1.0
    chunk_size: int = 20000
    output_dir: str = "output"
    base_url: str = "https://api.openai.com"
    api_key: str = None
    extra_context: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)

    def safe_dict(self):
        d = self.model_dump()
        d["api_key"] = d["api_key"][:5]
        return d


class QualitativeAnalysis(BaseModel):
    themes: Optional[List[Theme]] = None
    codes: Optional[List[Code]] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[dict] = None

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
