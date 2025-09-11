rm -rf dist build *.egg-info
uv build  && twine check dist/*
uv pip install -e .
uv run chatter "Tell me a joke: [[joke]]. Was it funny? [[bool:funny]]"