FROM python:3.11-slim
LABEL authors="elbekbakiev"

# Install build dependencies (gcc, make, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the backend files into the container
COPY ./backend/ /app/

RUN pip install --upgrade pip
RUN pip install pesq
RUN pip install -r requirements.txt

# Expose the port FastAPI is running on
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
