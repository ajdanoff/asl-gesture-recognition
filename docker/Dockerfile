FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt /workspace/requirements-ci.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements-ci.txt

COPY scripts /workspace/scripts
COPY .github /workspace/.github

RUN mkdir -p /workspace/benchmark \
    /workspace/benchmark-results \
    /workspace/benchmark-reports

WORKDIR /workspace

ENTRYPOINT ["python"]