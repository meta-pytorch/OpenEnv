ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e /app
RUN pip install --no-cache-dir "gradio>=4.0.0"

ENV PYTHONPATH=/app/src:/app/envs:/app
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
