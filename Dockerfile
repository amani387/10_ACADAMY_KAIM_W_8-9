# Use an official Python runtime as base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt
## Build the Docker image
#docker build -t fraud-detection-api .

# Run the container on port 5000
#docker run -p 5000:5000 fraud-detection-api

# Expose API port
EXPOSE 5000

# Run the Flask application
CMD ["python", "serve_model.py"]
