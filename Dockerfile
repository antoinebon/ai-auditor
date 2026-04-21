# syntax=docker/dockerfile:1.7

# --- Stage 1: builder ---------------------------------------------------
# Pull deps with uv in a fat image so we have the build toolchain, then
# we copy only the resolved virtualenv into the slim runtime stage.
FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv by pinned version for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /usr/local/bin/uv

WORKDIR /app

# Resolve dependencies first so we get a cached layer that only busts when
# pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Now bring in the source and install the project itself into the venv.
COPY src ./src
COPY data ./data
COPY README.md ./
RUN uv sync --frozen

# --- Stage 2: runtime ---------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    OLLAMA_HOST="http://host.docker.internal:11434" \
    OLLAMA_MODEL="qwen2.5:7b-instruct" \
    CONTROLS_PATH="/app/data/controls/iso27001_annex_a.yaml"

# pymupdf wheels are statically linked on Linux, so no runtime shared libs
# are needed beyond what python:3.12-slim ships.
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/data /app/data
COPY --from=builder /app/README.md /app/README.md

# The container needs to reach the host's Ollama; on Linux the caller
# should pass --add-host=host.docker.internal:host-gateway, otherwise
# override OLLAMA_HOST at run time.
ENTRYPOINT ["ai-auditor"]
CMD ["--help"]
