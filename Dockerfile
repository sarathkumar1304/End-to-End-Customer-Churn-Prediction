# Dockerfile for all services
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose port
EXPOSE 8001 
EXPOSE 5000 
EXPOSE 8501  

