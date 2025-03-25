# Use the official Python image with slim dependencies
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for model execution
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Set environment variables
ENV PORT=8080

# Command to run the app
CMD ["python", "main.py"]
