# Define environment and Python version variables
SHELL := /bin/bash
PYTHON_VERSION := 3
ENV_NAME := env
DOCKER_IMAGE := my-churn-model
DOCKER_CONTAINER := churn-model-container
DOCKERFILE := dockerfile
DATA_VOLUME := churn_model_data

# Local (non-Docker) commands
create-env:
	( \
		rm -rf ${ENV_NAME} && \
		python${PYTHON_VERSION} -m venv ${ENV_NAME} && \
		source ${ENV_NAME}/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt \
	)

run-modeling:
	time ( \
		source ${ENV_NAME}/bin/activate && \
		export PYTHONPATH=`pwd` && \
		python run/run_churn_model.py \
	)

run-tests:
	time ( \
		source ${ENV_NAME}/bin/activate && \
		export PYTHONPATH=`pwd` && \
		pytest scripts/ -v \
	)

# Docker commands

docker-build:
	docker build -t ${DOCKER_IMAGE} -f ${DOCKERFILE} .

# Run the modeling inside Docker and save output to the named volume
docker-run-modeling:
	docker run --rm --name ${DOCKER_CONTAINER} \
		-v ${DATA_VOLUME}:/app ${DOCKER_IMAGE} run-modeling

# Run tests inside Docker using the data saved in the volume
docker-run-tests:
	docker run --name ${DOCKER_CONTAINER} \
		-v ${DATA_VOLUME}:/app ${DOCKER_IMAGE} run-tests

# Copy results from Docker volume to local "results_from_docker" folder
docker-copy-results:
	mkdir -p results_from_docker/logs results_from_docker/images results_from_docker/models
	docker cp ${DOCKER_CONTAINER}:/app/logs/. results_from_docker/logs/
	docker cp ${DOCKER_CONTAINER}:/app/images/. results_from_docker/images/
	docker cp ${DOCKER_CONTAINER}:/app/models/. results_from_docker/models/

# Stops and removes Docker containers with a specific name pattern and deletes the Docker image
docker-stop:
	docker stop ${DOCKER_CONTAINER} || true
	docker rm ${DOCKER_CONTAINER} || true
	docker rmi ${DOCKER_IMAGE} || true
	docker volume rm ${DATA_VOLUME} || true

# Run all steps locally
local-full-run: create-env run-modeling run-tests

# Run all steps in Docker with the volume
docker-full-run: docker-build docker-run-modeling docker-run-tests docker-copy-results docker-stop
