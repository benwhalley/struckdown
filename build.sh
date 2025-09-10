rm -rf dist build *.egg-info
uv build  && twine check dist/*
uv pip install -e .