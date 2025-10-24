# Dockerfile (root)
FROM python:3.11-slim
WORKDIR /app

# Faster, cleaner installs
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add source
COPY . .

# App listens on 80
EXPOSE 80

# ---- Choose ONE of the following commands ----
# FastAPI (if your app object is at app.py -> app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]