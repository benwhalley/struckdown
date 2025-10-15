import logging
from typing import List, Optional

import typer
from decouple import config as env_config

from . import ACTION_LOOKUP, LLM, LLMCredentials, chatter

logging.getLogger("chatter").setLevel(logging.WARNING)

app = typer.Typer()


@app.command()
def run(
    prompt: List[str] = typer.Argument(
        ..., help="Prompt with slots, e.g. tell a joke [[joke]]"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    show_context: bool = typer.Option(False, help="Print the resolved prompt context"),
    
    verbose: bool = typer.Option(False, help="Print the full ChatterResult object"),
):
    """
    Run the chatter pipeline from the command line.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    prompt_str = " ".join(prompt)  # Join tokens into single prompt
    credentials = LLMCredentials()

    model = LLM(model_name=model_name)
    result = chatter(
        multipart_prompt=prompt_str,
        model=model,
        credentials=credentials,
        action_lookup=ACTION_LOOKUP,
    )

    for k, v in result.results.items():
        typer.echo(f"{k}: {v.output}")

    if show_context:
        typer.echo("\nFinal context:")
        typer.echo(result.outputs)
    

if __name__ == "__main__":
    app()
