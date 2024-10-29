# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV ENV_NAME=env

# Set the working directory
WORKDIR /app

# Copy the requirements file and Makefile into the container
COPY requirements.txt ./
COPY makefile ./

# Install required packages and Make
RUN apt-get update && \
    apt-get install -y make && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the entry point to make commands
ENTRYPOINT ["make"]
