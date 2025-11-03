ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY . /app

CMD ["uvicorn", "envs.__ENV_NAME__.server.app:app", "--host", "0.0.0.0", "--port", "8000"]


