# AstrocyteX image (CPU-portable, pre-trained checkpoints included).
# Build & run:
#   docker build -t astrocytex .
#   docker run --rm -p 8501:8501 astrocytex
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU-only PyTorch keeps the image compact and portable across all host machines.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
COPY . .
EXPOSE 8501 8000
CMD ["python", "run.py", "ui"]
