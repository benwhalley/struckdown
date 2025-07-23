
# llemma: tools for psychological research, education and practice

- chatter
- soak



# SETUP 

Requires 2 env vars to be set:

LLM_API_KEY
LLM_ENDPOINT

Then setup an alias:

```
docker build -t soak .
alias soak='docker run -it \
  -e LLM_API_KEY \
  -e LLM_ENDPOINT \
  -v "$HOME/.uv-cache":/root/.cache/uv \
  -v "$PWD":/app \
  -w /app soak uv run soak'
```

And finally:

soak run --help 

```
soak run demo data/5LC.docx
```