# MLDoctorEnv — runs on free HF CPU Space (vcpu=2, memory=8GB)

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY mldoctor_env /app/mldoctor_env
COPY openenv.yaml /app/openenv.yaml
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN pip install --no-cache-dir -e .

ENV PORT=7860
EXPOSE 7860
ENV MLDOCTOR_TASK=obvious_failure_diagnosis

CMD ["uvicorn", "mldoctor_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
