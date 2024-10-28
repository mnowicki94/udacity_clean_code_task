SHELL := /bin/bash
PYTHON_VERSION_FILE := 3.10
PYTHON_VERSION := $(file < $(PYTHON_VERSION_FILE))
ENV_NAME := env_udacity_clean_code_task

create-env:
	( \
		rm -rf ${ENV_NAME}  && \
		python3 -m venv ${ENV_NAME}  && \
		source ${ENV_NAME}/bin/activate  && \
		pip3 install --upgrade pip  && \
		pip3 install -r requirements.txt  \
	)


run-modeling:
	time ( \
		source ${ENV_NAME}/bin/activate  && \
		export PYTHONPATH=`pwd` && \
		python run/run_churn_model.py  \
	)

run-tests:
	time ( \
		source ${ENV_NAME}/bin/activate  && \
		export PYTHONPATH=`pwd` && \
		pytest scripts/ -v  \
	)
