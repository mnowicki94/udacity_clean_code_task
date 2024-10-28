# Define environment and Python version variables
SHELL := /bin/bash
PYTHON_VERSION_FILE := 3.10
PYTHON_VERSION := $(shell cat $(PYTHON_VERSION_FILE))
ENV_NAME := env

# Local (non-Docker) commands

# Creates the Python virtual environment locally
create-env:
	( \
		rm -rf ${ENV_NAME} && \
		python${PYTHON_VERSION} -m venv ${ENV_NAME} && \
		source ${ENV_NAME}/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt \
	)

# Run the churn model locally
run-modeling:
	time ( \
		source ${ENV_NAME}/bin/activate && \
		export PYTHONPATH=`pwd` && \
		python run/run_churn_model.py \
	)

# Run tests locally
run-tests:
	time ( \
		source ${ENV_NAME}/bin/activate && \
		export PYTHONPATH=`pwd` && \
		pytest scripts/ -v \
	)

# Docker-related variables
IMAGE_NAME := udacity_churn_model
CONTAINER_NAME := udacity_churn_model_container

# Docker commands

# Builds the Docker image
docker-build:
	docker build --no-cache -t ${IMAGE_NAME} .

# Runs the model inside Docker
docker-run-modeling:
	docker run --name ${CONTAINER_NAME} ${IMAGE_NAME} make run-modeling
    # docker run --rm --name ${CONTAINER_NAME} ${IMAGE_NAME} make run-modeling
    # would automatically close container after finish

# Runs the tests inside Docker
docker-run-tests:
	docker run --name ${CONTAINER_NAME} ${IMAGE_NAME} make run-tests

# Copies results from Docker container to local "results_from_docker" folder
docker-copy-results:
	mkdir -p results_from_docker
	docker cp ${CONTAINER_NAME}:/udacity_clean_code_task/images results_from_docker/images
	docker cp ${CONTAINER_NAME}:/udacity_clean_code_task/models results_from_docker/models
	docker cp ${CONTAINER_NAME}:/udacity_clean_code_task/logs results_from_docker/logs

# Stops the Docker container
docker-stop:
	docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}
