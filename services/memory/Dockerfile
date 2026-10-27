FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG TARGETARCH=amd64

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONPATH=/app PATH=/opt/scip-python-env/bin:/app/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git nodejs npm && npm install --global @sourcegraph/scip-python@0.6.6 && uv venv --seed --python 3.12 /opt/scip-python-env && curl -fsSL "https://github.com/scip-code/scip/releases/download/v0.10.0/scip-linux-${TARGETARCH}.tar.gz" -o /tmp/scip.tar.gz && tar -xzf /tmp/scip.tar.gz -C /usr/local/bin scip && chmod +x /usr/local/bin/scip && rm -rf /tmp/scip.tar.gz /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY services/memory/ services/memory/
COPY services/workflow/ services/workflow/

RUN uv sync --frozen --package agentteam-memory --no-dev

EXPOSE 8001
CMD ["uvicorn", "services.memory.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
