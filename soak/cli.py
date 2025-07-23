"""Command-line interface for running qualitative analysis pipelines."""

import json
import os
import sys
from pathlib import Path

import typer
from decouple import config as env_config

from .dag import pipeline_from_yaml
from .document_utils import extract_text, unpack_zip_to_temp_paths_if_needed

import logging


def get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1_000):
        new_path = path.with_stem(f"{path.stem}_{i}")
        if not new_path.exists():
            return new_path
    raise FileExistsError("Too many files with the same name.")


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
    output: str = typer.Option("output.json", "--output", "-o"),
):
    """Run a pipeline on input files."""

    try:
        pipyml = PIPELINE_DIR / pipeline / "soak.yaml"
        if not pipyml.is_file():
            raise FileNotFoundError(f"No such default pipeline")
    except FileNotFoundError:
        pipyml = Path(pipeline)
        if not pipyml.is_file():
            raise FileNotFoundError(f"Pipeline file not found: {pipyml}")
    pipeline = pipeline_from_yaml(pipyml.read_text())
    pipeline.document_paths = None
    pipeline._documents = None

    print(pipeline)

    with unpack_zip_to_temp_paths_if_needed(input_files) as docfiles:
        pipeline.document_paths = docfiles

    try:
        # all pipelines are now DAG pipelines
        result = pipeline.run()
    except Exception as e:
        raise typer.Exit(1)

    np = get_unique_path(Path(output))
    print(f"Writing output to {np}")
    with open(Path(np), "w") as f:
        f.write(json.dumps(result.result().model_dump()))


@app.command(name="list")
def list_pipelines():
    """List available DAG pipelines."""
    yaml_files = find_pipeline_yaml_files()

    if yaml_files:
        print("Available DAG pipelines:")
        for path in yaml_files:
            try:
                _ = pipeline_from_yaml(path.read_text())
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
