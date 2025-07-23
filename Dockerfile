FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y libmagic-dev

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install deps (this layer is cached as long as lockfile doesn't change)
RUN uv sync

# Copy rest of the code
COPY . .

# Default command
CMD ["uv", "run"]
