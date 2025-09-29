# Use an official Python runtime as a base image
FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (optional: for psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (FastAPI default: 8000)
EXPOSE 8000

# Run the application with Uvicorn
# Replace "main:app" with your FastAPI entrypoint (filename:app_instance)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
