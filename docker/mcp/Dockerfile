FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        "mcp>=1.18.0" \
        "pydantic>=2.0,<3.0" \
        "roslibpy>=0.3.0,<2.0.0"

COPY server/moveit_mcp ./moveit_mcp
COPY server/vizor_mcp ./vizor_mcp

CMD ["python", "-m", "vizor_mcp", "--help"]
