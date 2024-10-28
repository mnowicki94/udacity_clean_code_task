# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory in the container
WORKDIR /udacity_clean_code_task

COPY requirements.txt ./
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/* && \
    python -m venv env && \
    ./env/bin/pip install --upgrade pip && \
    ./env/bin/pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /udacity_clean_code_task

# Set the environment variable PATH to use the virtual environment
ENV PATH="/udacity_clean_code_task/env/bin:$PATH"

# Define the default command to keep the container running and allow Make commands
CMD ["make"]
