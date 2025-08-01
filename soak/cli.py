"""Command-line interface for running qualitative analysis pipelines."""

import json
import logging
import os
import sys
from pathlib import Path

import typer
import yaml
from chatter import LLMCredentials
from decouple import config as env_config

from .document_utils import unpack_zip_to_temp_paths_if_needed
from .specs import load_template_bundle

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("chatter").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger().setLevel(logging.INFO)


sys.path.append(os.path.dirname(os.path.dirname(__file__)))


PIPELINE_DIR = Path(__file__).parent / "pipelines"


def find_pipeline_yaml_files():
    return list(PIPELINE_DIR.rglob("*.yaml"))


def pipeline_name_from_path(path: Path) -> str:
    return str(path.relative_to(PIPELINE_DIR).with_suffix(""))  # strip .yaml


app = typer.Typer()


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'poc')"),
    input_files: list[str] = typer.Argument(
        ..., help="File patterns or zip files (supports globs like '*.txt')"
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output file path (stdout if not specified)"
    ),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, yaml, or html"),
    include_documents: bool = typer.Option(
        False, "--include-documents", help="Include original documents in output"
    ),
):
    """Run a pipeline on input files."""

    # validate format parameter
    if format not in ["json", "yaml", "html"]:
        raise typer.BadParameter("Format must be 'json', 'yaml', or 'html'")

    try:
        pipyml = PIPELINE_DIR / pipeline / "soak.yaml"
        if not pipyml.is_file():
            raise FileNotFoundError(f"No such default pipeline")
    except FileNotFoundError:
        pipyml = Path(pipeline)
        if not pipyml.is_file():
            raise FileNotFoundError(f"Pipeline file not found: {pipyml}")

    print(f"Loading pipeline from {pipyml}", file=sys.stderr)
    pipeline = load_template_bundle(pipyml)

    pipeline.config.document_paths = None
    pipeline.config.documents = None

    pipeline.config.llm_credentials = LLMCredentials()

    with unpack_zip_to_temp_paths_if_needed(input_files) as docfiles:
        pipeline.config.document_paths = docfiles

    try:
        # all pipelines are now DAG pipelines - run directly with asyncio
        import asyncio

        analysis = asyncio.run(pipeline.run())
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)

    # import pdb; pdb.set_trace()

    analysis.config.llm_credentials.llm_api_key = (
        analysis.config.llm_credentials.llm_api_key[:5] + "***"
    )

    # remove documents from output if not requested
    if not include_documents:
        analysis.config.documents = []

    # generate output content based on format
    if format == "json":
        content = json.dumps(analysis.model_dump(), indent=2)
    elif format == "yaml":
        content = yaml.dump(analysis, default_flow_style=False, indent=2)
    elif format == "html":
        content = analysis.to_html()

    # output to stdout or file
    if output is None:
        print(content)
    else:
        print(f"Writing output to {output}")
        with open(output, "w") as f:
            f.write(content)


@app.command(name="list")
def list_pipelines():
    """List available DAG pipelines."""
    yaml_files = find_pipeline_yaml_files()

    if yaml_files:
        print("Available DAG pipelines:")
        for path in yaml_files:
            try:
                _ = load_template_bundle(path)
                status = "✓"
                issues = []
            except Exception as e:
                raise e
                status = "✗"
                issues = [str(e)]

            name = pipeline_name_from_path(path)
            print(f"  {status} {name}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("No DAG pipelines found.")


if __name__ == "__main__":
    app()
