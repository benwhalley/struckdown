# TODO

## Grammar: allow hyphens in positional `[[pick:...]]` options

`struckdown/grammar.lark` defines positional pick options via lark's `CNAME`
token (Python-identifier shape: `[_A-Za-z][_A-Za-z0-9]*`). Hyphenated slugs
like `writing-concisely` tokenise as `CNAME('writing')` and then fail on
`-concisely`, breaking templates like
`[[pick:task|writing-concisely,mean-marker,...]]`.

Slugs / kebab-case ids are common, so positional options should accept them
natively. Add a `SLUG` token (`("_"|"-"|LETTER) ("_"|"-"|LETTER|DIGIT)*`) and
extend `positional_value` to emit it as a literal string:

```
positional_value: STRING -> literal_string
                | SIGNED_FLOAT -> literal_float
                | NUMBER -> literal_number
                | EMOJI -> literal_emoji
                | SLUG -> literal_slug   # new -- accepts hyphens
                | CNAME -> literal_cname
```

Workaround until then: quote each option (`[[pick:task|"writing-concisely",...]]`).
psybot/record/ai.py applies this workaround in `match_task2`.

## Test `ThinkingDelta` against a local thinking model

`ThinkingDelta` (0.12.0) is wired but has never streamed a real token. The
handler is unit-tested (`tests/test_tool_slots.py::ThinkingStreamTests`) by
feeding it pydantic-ai part events directly, because attaching an
`event_stream_handler` makes pydantic-ai stream the request, which its own
`FunctionModel` cannot fake without a `stream_function`.

Nothing end to end has produced one. Every model available through the WAM
LiteLLM proxy -- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.6-sol` -- returns only
`ToolCallPart` over chat-completions: OpenAI keeps raw reasoning server-side,
so there is no thinking to stream.

**Ollama closes that gap without a provider, a key or a bill.** The harness
already exists: `tests/test_ollama_local.py` finds a local qwen3 and skips when
there is none, and qwen3 is a thinking model whose reasoning comes back in the
response rather than being withheld.

```bash
ollama pull qwen3:8b
uv run pytest struckdown/tests/test_ollama_local.py -q
```

What to add there, marked `requires_ollama`:

- a tool slot against qwen3 emits `ThinkingDelta`, and `accumulated` grows
  monotonically across them;
- thinking arrives *between* tool calls, not only before the answer -- the
  claim the feature is sold on, and the one nothing currently checks;
- `SlotResult.completion["_thinking_steps"]` holds one entry per step rather
  than only the last, which is what the per-step collection in
  `_completion_dict_for_run` exists for;
- a tripped `[[halt:...]]` still retracts cleanly when thinking is mid-stream.

Worth doing before anyone points a reasoning-content model at this in
production: the display path in WAM (`tower/qa/agent_loop.py`) and the Hub
(`hub_assistant/loop.py`) has also never rendered a real one.
