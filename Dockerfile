FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces handles external ports, but runs your app on 7860
CMD ["uvicorn", "backend_server:app", "--host", "0.0.0.0", "--port", "7860"]
