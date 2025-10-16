
# struckdown:  markdown-like syntax for structured conversations with language models

Markdown-based syntax for structured LLM use.

# TLDR;

Imagine you have some product data:

```bash
% ls *.txt
butter_robot.txt
meeseeks_box.txt
microverse_battery.txt
plumbus.txt
portal_gun.txt

% tail *.txt
==> plumbus.txt <==
First, they take the dinglebop and smooth it out with a bunch of schleem. The schleem is then repurposed for later batches. They take the dinglebop and push it through the grumbo, where the fleeb is rubbed against it. It's important that the fleeb is rubbed, because the fleeb has all of the fleeb juice. Then a Shlami shows up and he rubs it and spits on it. They cut the fleeb. There are several hizzards in the way.
The blamfs rub against the chumbles. And the ploobis and grumbo are shaved away. That leaves you with a regular old Plumbus!
Price: 6½ brapples
Dimensions: Standard Plumbus sizing
Weight: 2.3 greebles
Made in: Dimension C-137
```

The texts don't make much sense, but you check if the LLM knows better:

```
% sd batch *.txt "Purpose, <5 words: [[purpose]]"
[
  {
    "filename": "butter_robot.txt",
    "purpose": "Pass butter, question existence."
  },
  {
    "filename": "meeseeks_box.txt",
    "purpose": "Instant help for simple tasks."
  },
  {
    "filename": "microverse_battery.txt",
    "purpose": "Infinite spaceship power source"
  },
  {
    "filename": "plumbus.txt",
    "purpose": "Purpose, <5 words: Household universal utility device."
  },
  {
    "filename": "portal_gun.txt",
    "purpose": "Purpose: Interdimensional travel device."
  }
]
```


Or 

```
% sd batch *.txt "Price: [[number:price]] Currency? [[pick:currency|schmeckles,brapples,flurbos]]"
[
  {
    "filename": "butter_robot.txt",
    "price": 18,
    "currency": "schmeckles"
  },
  {
    "filename": "meeseeks_box.txt",
    "price": 45,
    "currency": "schmeckles"
  },
  {
    "filename": "microverse_battery.txt",
    "price": 850,
    "currency": "flurbos"
  },
  {
    "filename": "plumbus.txt",
    "price": 6.5,
    "currency": "brapples"
  },
  {
    "filename": "portal_gun.txt",
    "price": 10000000,
    "currency": "flurbos"
  }
]
```


`batch` accepts json as input, so you can even chain commands. 
If you want to work out the next best thing to buy on Amazon, you can do this:

```
% sd batch *.txt \
  "Purpose, <5 words: [[purpose]] Name [[name]]" | \
  sd batch \
    "{{name}} {{purpose}}. Most similar on amazon? Best guess, < 10 words [[product]]" -k
[
  {
    "filename": "butter_robot.txt",
    "purpose": "Pass butter, question existence.",
    "name": "Butter-Passing Robot",
    "product": "Rick and Morty \"Pass the Butter\" Robot Figure—Amazon search."
  },
  {
    "filename": "meeseeks_box.txt",
    "purpose": "Instant help for simple tasks.",
    "name": "Name: Mr. Meeseeks Box\nPurpose: Instant help for simple tasks.",
    "product": "Most similar on Amazon: \"Meeseeks Box Replica Toy\"."
  },
  {
    "filename": "microverse_battery.txt",
    "purpose": "Infinite spaceship power source",
    "name": "MICROVERSE BATTERY",
    "product": "Amazon closest match: \"Portable Solar Generator Power Station\"."
  },
  {
    "filename": "plumbus.txt",
    "purpose": "Purpose, <5 words: Household universal utility device.",
    "name": "Name: Plumbus",
    "product": "Most similar on Amazon: Multi-purpose cleaning gadget, 8-in-1 tool."
  },
  {
    "filename": "portal_gun.txt",
    "purpose": "Purpose: Interdimensional travel device.",
    "name": "Portal Gun",
    "product": "Rick and Morty Portal Gun Toy Replica"
  }
]
```


You can test prompts with the chat command:

```
% sd chat "Tell me a joke: [[joke]]"
```


# SETUP

Install UV: https://docs.astral.sh/uv/getting-started/installation

```
uv tool install https://github.com/benwhalley/struckdown/

# or 
uv pip install git+https://github.com/benwhalley/struckdown/  
```


Set environment variables:

```
export LLM_API_KEY=... # e.g. from openai.com
export LLM_API_BASE=... # e.g. https://api.openai.com/v1
export DEFAULT_LLM="litellm/gpt-4.1-mini"
```


## Cacheing

Struckdown caches LLM responses to disk to save costs and improve performance. 
The cache can be configured via environment variables:


`STRUCKDOWN_CACHE`

Controls the cache directory location:

- **Default**: `~/.struckdown/cache` or ~/$cwd/.struckdown/cache if that dir can't be created
- **Disable caching**: Set to `"0"`, `"false"`, or empty string
- **Custom location**: Set to any valid directory path



`STRUCKDOWN_CACHE_SIZE`

Controls the maximum cache size in **megabytes**:
- **Default**: `10240` (10 GB)
- **Unlimited**: Set to `0` (not recommended for production)
- When the limit is exceeded, the oldest cached items are automatically evicted (LRU policy)


Note: The cache is shared across all processes that use struckdown. 




# Detailed syntax guide

Prompts for steps and judgements are written in a "markdownish" format with extensions to specify completion types.

As part of a single prompt template, we can ask the AI to respond multiple times. Each new response forms part of the context for subsequent prompts.

Each response is specifed by a `[[RESPONSE]]` tag:

Optionally, a prefix can be used to guide the style of the AI response.
Presently `think` and `speak` are supported, but more may be added:

```
[[think:response]]
[[speak:response]]
```

A `think` response will be more reflective, longer, and can include notes/plans.

The `speak` response will be more direct, and the AI is requested to use spoken idioms. These different styles of responses are achieved by adding hints to the call to the AI model, and changing parameters like temperature.


#### Classifications

Two prefixes are supported to allow for classifications:

```
[[boolean:response]]
```

And

```
choose an option [[pick:color|red,green,blue]]
```


- `pick` guarantees that the response is one of the options provided, after the `|` character, separated by commas.
- `boolean` guarantees that the response is either True or False.


A multiline version of `pick` is also allowed:

```
[[pick:response
    option1
    option2
    option3
    default=null]]
```


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