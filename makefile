SHELL := /bin/bash
PYTHON_VERSION_FILE := 3.10
PYTHON_VERSION := $(file < $(PYTHON_VERSION_FILE))
ENV_NAME := env_udacity_clean_code_task

create-env:
	( \
		rm -rf ${ENV_NAME}  && \
		python${PYTHON_VERSION} -m venv ${ENV_NAME}  && \
		source ${ENV_NAME}/bin/activate  && \
		pip${PYTHON_VERSION} install --upgrade pip  && \
		pip${PYTHON_VERSION} install -r requirements.txt  \
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
