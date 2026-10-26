FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONPATH=/app PATH=/app/.venv/bin:$PATH

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY services/memory/ services/memory/
COPY services/workflow/ services/workflow/

RUN uv sync --frozen --package agentteam-memory --no-dev

EXPOSE 8001
CMD ["uvicorn", "services.memory.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
