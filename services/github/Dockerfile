FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONPATH=/app PATH=/app/.venv/bin:$PATH

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY services/memory/pyproject.toml services/memory/pyproject.toml
COPY services/workflow/pyproject.toml services/workflow/pyproject.toml
COPY services/github/ services/github/

RUN uv sync --frozen --package agentteam-github --no-dev

EXPOSE 8003
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/healthz', timeout=3)"]
CMD ["hypercorn", "services.github.app.main:app", "--bind", "0.0.0.0:8003"]
