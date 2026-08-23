FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository
COPY . .

# Change working directory to backend so that relative imports and DB paths work correctly
WORKDIR /app/backend

# Expose port 7860 which is the default for Hugging Face Spaces Docker
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
