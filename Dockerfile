# One image serves both the API and the built React app (used for the public demo on Render).
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

FROM python:3.11-slim
# g++ lets PyMC's backend (PyTensor) compile the sampler, which is much faster than its pure-Python fallback
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY expplat expplat
COPY sample_data sample_data
COPY --from=web /web/dist frontend/dist
RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV PYTENSOR_FLAGS="base_compiledir=/tmp/pytensor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
EXPOSE 7860
# hosts such as Render tell the app which port to use through $PORT
CMD ["sh", "-c", "uvicorn expplat.main:app --host 0.0.0.0 --port ${PORT:-7860}"]