# syntax=docker/dockerfile:1

# ---- Stage 1: build the React dashboard ----------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the FastAPI app, serving the built dashboard ----------------
# Matches the Python version this project has actually been developed and tested against.
FROM python:3.14-slim AS runtime
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Persisted outside the image so data survives container restarts/rebuilds —
# see the `memtrace_data` volume in docker-compose.yml.
ENV MEMTRACE_DB_PATH=/data/memtrace.db
RUN mkdir -p /data
VOLUME ["/data"]

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
