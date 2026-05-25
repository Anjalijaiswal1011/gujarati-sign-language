# Use python 3.10 slim as base image
FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to the app root
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY backend/requirements.txt /app/backend/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the rest of the application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Set working directory to backend where manage.py is located
WORKDIR /app/backend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DEBUG=True

# Expose port
EXPOSE 8000

# Start Daphne (ASGI server for Django WebSockets)
CMD sh -c "python manage.py migrate --noinput && daphne -b 0.0.0.0 -p ${PORT:-8000} core.asgi:application"
