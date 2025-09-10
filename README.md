
# llemma: tools for psychological research, education and practice

This project provides a markdown-based language for templated, multi-part LLM calls.



# SETUP

On OS X or linux:

Install UV: https://docs.astral.sh/uv/getting-started/installation



```
uv pip install llemma
```

Or

```
uv pip install git+https://github.com/benwhalley/llemma/  
uv pip install -e .
```


Set environment variables:

```
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...
export DEFAULT_LLM="litellm/gpt-4.1-mini"
```

Test the setup by running a struckdown prompt:

```
uv run chatter "Tell me a joke: [[joke]]. Was it funny? [[bool:funny]]"
```



