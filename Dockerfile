FROM public.ecr.aws/docker/library/python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY bedrock_connector ./bedrock_connector
COPY examples ./examples
RUN pip install --no-cache-dir -e . pytest
ENV BEDROCK_MODE=auto AGENT_WORKDIR=/tmp/runs
EXPOSE 8080
CMD ["bedrock-connector", "serve", "--host", "0.0.0.0", "--port", "8080"]
