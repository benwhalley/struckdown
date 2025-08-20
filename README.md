
# llemma: tools for psychological research, education and practice

- chatter
- soak


# SETUP

On OS X or linux:

Install UV

https://docs.astral.sh/uv/getting-started/installation


Clone the repo:

```
git clone https://github.com/benwhalley/llemma
cd llemma
```


Install the package:

```
uv pip install -e .
```

Set 2 environment variables:

```
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://your-endpoint.com (any OpenAI compatible)
```


# Running  with uv (recommended) 

## Chatter

Test the setup by running a chatter prompt:

```
uv run chatter "Tell me a joke: [[joke]]. Was it funny? [[bool:funny]]"
```


## Soak: qualitative analysis

```
uv run soak run demo soak/data/yt-cfs.txt --output yt-cfs-example
```



