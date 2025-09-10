FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y libmagic-dev
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync
COPY . .
CMD ["bash"]
