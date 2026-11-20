ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
# Override only with the official PyTorch wheel index appropriate to the host.
ARG TORCH_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir numpy==2.5.3 && python -m pip install --no-cache-dir torch==2.14.0 --index-url ${TORCH_INDEX_URL}
COPY pyproject.toml run.py /app/
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY docs /app/docs
COPY scripts /app/scripts
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "run.py"]
CMD ["doctor", "--device", "cuda"]
