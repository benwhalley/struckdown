
# With docker (on windows if uv fails)

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
chatter "tell me a joke [[joke]] was it funny? [[bool:funny]]"
soak run demo soak/data/yt-cfs.txt
```