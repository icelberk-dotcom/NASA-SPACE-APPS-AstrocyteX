# AstroGuard-X image (CPU). Build:  docker build -t astroguard-x .
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU-only PyTorch keeps the image small; remove the first line to use the default (CUDA) wheels.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
COPY . .
# Train the compact model at build time so the container starts instantly (simulated data, ~3-5 min on CPU).
RUN python -m models.train --samples 2500 --epochs 8
EXPOSE 8000 8501
CMD ["python", "run.py", "api"]
