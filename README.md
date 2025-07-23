
# llemma: tools for psychological research, education and practice

- chatter
- soak



# SETUP

Install UV

https://docs.astral.sh/uv/getting-started/installation


Set 2 env vars:

export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://your-endpoint.com (any OpenAI compatible)

Or on windows,

set LLM_API_KEY=your_api_key
set LLM_BASE_URL=https://your-endpoint.com


## Running with uv (recommended)


```
uv run chatter "tell me a joke [[joke]] was it funny? [[bool:funny]]"
uv run soak run demo data/5LC.docx
```

## Or with docker (on windows if uv fails)

```
docker build -t soak .
alias soak='docker run -it \
  -e LLM_API_KEY \
  -e LLM_BASE_URL \
  -v "$HOME/.uv-cache":/root/.cache/uv \
  -v "$PWD":/app \
  -w /app soak uv run soak'

alias chatter='docker run -it \
  -e LLM_API_KEY \
  -e LLM_BASE_URL \
  -v "$HOME/.uv-cache":/root/.cache/uv \
  -v "$PWD":/app \
  -w /app soak uv run chatter'


```

Then

```
soak run demo data/5LC.docx
chatter "tell me a joke [[joke]] was it funny? [[bool:funny]]"
```