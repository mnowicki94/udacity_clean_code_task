# Define environment and Python version variables
SHELL := /bin/bash
PYTHON_VERSION := 3.10
ENV_NAME := env
DOCKER_IMAGE := my-churn-model
DOCKERFILE := dockerfile
CONTAINER_NAME := churn-model-container

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

# Docker commands

# Build the Docker image
docker-build:
	docker build -t ${DOCKER_IMAGE} -f ${DOCKERFILE} .

# Create the environment inside Docker
docker-create-env:
	docker run --rm --name ${CONTAINER_NAME} ${DOCKER_IMAGE} create-env

# Run the modeling inside Docker
docker-run-modeling:
	docker run --rm --name ${CONTAINER_NAME} ${DOCKER_IMAGE} run-modeling

# Run tests inside Docker
docker-run-tests:
	docker run --name ${CONTAINER_NAME} ${DOCKER_IMAGE} run-tests

# Copy results from Docker container to local "results_from_docker" folder
docker-copy-results:
	mkdir -p results_from_docker
	docker cp ${CONTAINER_NAME}:/app/images results_from_docker/images
	docker cp ${CONTAINER_NAME}:/app/models results_from_docker/models
	docker cp ${CONTAINER_NAME}:/app/logs results_from_docker/logs

# Stops the Docker container and clean image
docker-stop:
	docker stop ${CONTAINER_NAME} || true
	docker rm ${CONTAINER_NAME} || true
	docker rmi ${DOCKER_IMAGE} || true


# Run all steps in Docker
docker-full-run: docker-build docker-create-env docker-run-modeling docker-run-tests docker-copy-results docker-stop

# Run all steps locally
local-full-run: create-env run-modeling run-tests