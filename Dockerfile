# --- Stage 1: build the React dashboard ---
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: python runtime serving API + static frontend ---
FROM python:3.12-slim
WORKDIR /app

# libgomp1 is required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
# Copy only needed model files (skip 30MB thin_file_tabpfn.joblib — TabPFN disabled)
COPY models/hazard.joblib models/calibrators.joblib models/confidence_bands.joblib models/
COPY models/*.json models/
COPY data/processed/serving_features.parquet data/processed/
COPY data/processed/serving_features_enriched.parquet data/processed/
COPY data/processed/phase4_scored_test.parquet data/processed/
COPY --from=frontend /fe/dist frontend/dist

EXPOSE 8000
# $PORT is injected by Render/Heroku-style hosts; defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
