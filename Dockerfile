FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS api-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MODEL_URI=/models/als

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-api.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

COPY src/__init__.py src/files.py ./src/
COPY src/api/ ./src/api/
COPY src/data/__init__.py src/data/id_mapper.py ./src/data/
COPY src/models/__init__.py src/models/collaborative.py src/models/registry.py src/models/serving.py ./src/models/

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Requests only need the film factors and the catalog: the serving copy leaves
# out the training users and matrix, about 80 % of a MovieLens 32M bundle.
FROM api-base AS serving-bundle
COPY --from=model catalog.json interactions.npz manifest.json model.npz /tmp/full/
RUN python -m src.models.serving /tmp/full /tmp/serving

FROM api-base AS deployment

# AWS Lambda serves this image through the web adapter extension, which stays
# inactive when the image runs anywhere else. With asynchronous initialisation,
# a start longer than Lambda's 10 s init limit finishes during the first request
# instead of restarting the whole environment.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0@sha256:17cfd08eff1dfea3f6a9a1e9c65fdac80aa4919b6085e746615530f43f57d2f1 \
    /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    AWS_LWA_ASYNC_INIT=true

USER 0:0
COPY --from=serving-bundle --chown=0:0 /tmp/serving/ /models/als/
RUN chmod 0555 /models /models/als \
    && chmod 0444 /models/als/*
USER 10001:10001
RUN python -c "from src.models.collaborative import ImplicitRecommender; ImplicitRecommender.load('/models/als')"

ARG VCS_REF
ENV APP_REVISION=${VCS_REF}
LABEL org.opencontainers.image.source="https://github.com/SamirMaoude/movies-and-me-recommender" \
    org.opencontainers.image.revision="${VCS_REF}"

FROM api-base AS runtime
