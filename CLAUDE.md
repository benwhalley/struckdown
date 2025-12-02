# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Struckdown: markdown-like syntax for structured LLM conversations. Uses `[[slot]]` for completions and `{{var}}` for references an [[@action]] for actions.

## Commands

- Run tests: `uv run python -m pytest struckdown/tests/ -v`
- Run single test: `uv run python -m pytest struckdown/tests/test_cache.py -v`
- CLI: `uv run sd chat "Tell a joke: [[joke]]"` or `uv run sd batch *.txt "[[summary]]"`

## Architecture

- `__init__.py`: Public API -- `chatter()` / `chatter_async()` process templates through segments
- `parsing.py`: Lark grammar parser (`grammar.lark`) transforms templates into PromptPart AST
- `segment_processor.py`: Delta-based processing -- re-renders Jinja after each slot completion
- `execution.py`: Dependency graph resolution and parallel LLM calls
- `sd_cli.py`: Typer CLI (`sd chat`, `sd batch`, `sd check`, `sd graph`, `sd flat`)
- `actions/`: Extensible action registry (`@action` syntax) -- see `docs/CUSTOM_ACTIONS.md`
- `types/`: YAML-defined response types (loaded by `type_loader.py`)
- `tools/`: Tool definitions (loaded by `tools_loader.py`)

## Key Syntax

- `[[type:var]]` -- typed completion slot (bool, number, date, pick, extract)
- `[[@action:var|params]]` -- action call (fetch, break, set, etc.)
- `<system>` / `<system local>` -- system messages (global vs segment-scoped)
- `<checkpoint>` -- memory boundary, only extracted vars carry forward
- `{% include 'file.sd' %}` -- Jinja2 includes

## Environment

- `LLM_API_KEY`, `LLM_API_BASE`, `DEFAULT_LLM` -- LLM configuration
- `STRUCKDOWN_CACHE=0` -- disable caching; `STRUCKDOWN_CACHE=/path` -- custom location
