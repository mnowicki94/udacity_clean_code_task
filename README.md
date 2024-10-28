# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This repository contains a Python project for predicting customer churn using machine learning. The project follows best clean code practice involving logging and unit tests.


## Files and data description
Overview of the files and data present in the root directory.

Project structure:
├── ./README.md
├── ./__init__.py
├── ./data
│   └── ./data/bank_data.csv
├── ./makefile
├── ./requirements.txt
├── ./run
│   ├── ./run/__init__.py
│   ├── ./run/constants.py
│   └── ./run/run_churn_model.py
└── ./scripts
    ├── ./scripts/__init__.py
    ├── ./scripts/churn_library.py
    ├── ./scripts/churn_script_logging_and_test.py
    └── ./scripts/conftest.py

Files description:
- run/:
    - constants.py: Defines constants used throughout the project, such as data paths and column names.
    - run_churn_model.py: The main script that runs the churn prediction pipeline.

- scripts/:
    - churn_library.py: Contains the core logic for data loading, preprocessing, model training, and prediction.
    - churn_script_logging_and_test.py: Contains unit tests for the churn_library.py module.
    - conftest: functions that are used by churn_script_logging_and_test.py.

- requirements.txt: A list of Python packages required for the project.

- makefile: Provides commands for setting up the project environment, running the churn prediction pipeline, and executing 
unit tests.
    - create-env: Creates a virtual environment for the project.
    - run-modeling: Runs the churn prediction pipeline.
    - run-tests: Runs unit tests.


- data/:
    - bank_data.csv - customers banking data

## Prerequisites

- **Python 3.10**: This project requires Python version 3.10. Ensure that it is installed on your system.
- **Makefile**: A Makefile is included in the project for automating tasks. Make sure you have `make` installed on your system.


## Running Files locally
Running steps are specified in makefile.

1. Create a virtual environment:
$ make create-env

2. Run the churn prediction pipeline:
$ make run-modeling

3. Run unit tests:
$ make run-tests

## Running Files in Docker
Running steps are specified in makefile.

1. Create a virtual environment:
$ make docker-build

2. Run the churn prediction pipeline:
$ make docker-run-modeling

3. Stop container:
$ make docker-stop

4. Run unit tests:
$ make docker-run-tests

5. Stop container:
$ make docker-stop

6. Copy results to local directory:
$ make docker-copy-results

7. Stop container:
$ make docker-stop