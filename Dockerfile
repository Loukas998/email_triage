# The eval runner, the golden set and the gate in one image. The model is not in it: Ollama runs on
# the host (or wherever OLLAMA_HOST points), because a 5 GB model does not belong in an image.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, in their own layer: they change far less often than the code.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

# Docker Desktop's name for the host machine. Override with -e OLLAMA_HOST=... anywhere else.
ENV OLLAMA_HOST=http://host.docker.internal:11434

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m"]
CMD ["email_triage.gate", "v1"]