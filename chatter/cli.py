import logging
from pathlib import Path
from typing import List

import typer

from . import ACTION_LOOKUP, LLM, LLMCredentials, chatter

logging.getLogger("prefect").setLevel(logging.WARNING)
logging.getLogger("chatter").setLevel(logging.WARNING)


app = typer.Typer()


@app.command()
def run(
    prompt: List[str] = typer.Argument(..., help="Prompt with slots, e.g. tell a joke [[joke]]"),
    model_name: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    show_context: bool = typer.Option(False, help="Print the resolved prompt context"),
):
    """
    Run the chatter pipeline from the command line.
    """
    prompt_str = " ".join(prompt)  # Join tokens into single prompt
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)
    result = chatter(prompt_str, model=model, credentials=credentials, action_lookup=ACTION_LOOKUP)

    for k, v in result.items():
        typer.echo(f"{k}: {v}")

    if show_context:
        typer.echo("\nFinal context:")
        typer.echo(result.outputs)


if __name__ == "__main__":
    app()
