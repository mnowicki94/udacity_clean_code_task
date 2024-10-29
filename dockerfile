# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV ENV_NAME=env
ENV PATH="/app/${ENV_NAME}/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and Makefile into the container
COPY requirements.txt ./
COPY makefile ./

# Install required packages and Make, create the virtual environment, and install dependencies
RUN apt-get update && \
    apt-get install -y make && \
    python -m venv ${ENV_NAME} && \
    /app/${ENV_NAME}/bin/pip install --upgrade pip && \
    /app/${ENV_NAME}/bin/pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of your application code
COPY . .

# Set the entry point to use Make commands by default
ENTRYPOINT ["make"]
