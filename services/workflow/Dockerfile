FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# The docker CLI talks to the host daemon over the mounted socket, so the
# sandbox can create containers that run the repository's tests.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
    | tar -xz -C /tmp \
 && mv /tmp/docker/docker /usr/local/bin/docker \
 && rm -rf /tmp/docker \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONPATH=/app PATH=/app/.venv/bin:$PATH

COPY pyproject.toml uv.lock ./
COPY agentteam.yaml ./
COPY packages/ packages/
COPY services/memory/ services/memory/
COPY services/workflow/ services/workflow/

RUN uv sync --frozen --package agentteam-workflow --no-dev

EXPOSE 8002
CMD ["hypercorn", "services.workflow.app.main:app", "--bind", "0.0.0.0:8002"]
