
# struckdown:  markdown based syntax for structured conversations with language models

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
export LLM_API_KEY=...
export LLM_API_BASE=...
export DEFAULT_LLM="litellm/gpt-4.1-mini"
```

Test the setup by running a struckdown prompt:

```
uv run chatter "Tell me a joke: [[joke]]. Was it funny? [[bool:funny]]"
```



# Detailed syntax guide

Prompts for steps and judgements are written in a simple markdown format with extensions to specify completion types.

As part of a single prompt template, we can ask the AI to respond multiple times. Each new response forms part of the context for subsequent prompts.

---


Each response is specifed by a `[[RESPONSE]]` tag:

Optionally, a prefix can be used to guide the style of the AI response.
Presently `think` and `speak` are supported, but more may be added:

```
[[think:response]]
[[speak:response]]
```

A `think` response will be more reflective, longer, and can include notes/plans.

The `speak` response will be more direct, and the AI is requested to use spoken idioms. (These different styles of responses are achieved by adding hints to the call to the AI model.)


#### Classifications

Two prefixes are supported to allow for classifications:

```
[[boolean:response]]
```

And

```
[[pick:response|option1, option2, option3, default=null]]
```


- `pick` guarantees that the response is one of the options provided, after the `|` character, separated by commas.
- `boolean` guarantees that the response is either True or False.

These are useful when making classifications, or for Judgements that determine whether a
step transition should take place.


A multiline version of `pick` is also allowed:

```
[[pick:response
    option1
    option2
    option3
    default=null]]
```



<!---
TODO: Implement this

Finally, advanced users can pass extra parameters to `[[response]]` slots, using the following syntax:

```
[[think:planning]]{max_tokens=50}

[[bool:is_upset]]{allow_null=true}
```
--->


#### Splitting prompts and saving tokens with `OBLIVIATE!`

Sometimes, we want to:

A. use an initial prompt to create a response
B. refine the response, using secondary instructions

In part A, we provide the LLM a lot of context.
In part B, we may not need all this context.

To save tokens, we can take the response from part A, and use it as input for part B.
This is done with the `¡OBLIVIATE` tag.

Example:

```
Long context about the history of vampires
Tell me a joke
[[speak:joke]]

¡OBLIVIATE

This is a joke:
{{joke}}

Tell me, is it funny:

[[boolean:funny]]
```

The key here is that when we are deciding if the joke is funny, we don't need the original context, so it's hidden. This speeds up generation and saves cost.




### Minimal example


```
Pick a fruit

[[pick|apple,orange,banana]]

Tell me a joke about your fruit

[[joke]]

¡OBLIVIATE

Tell me a joke about your job

[[joke2]]
```