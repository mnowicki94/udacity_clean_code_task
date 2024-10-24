"""
Author: Maciej N. 
Date Created: 2024-10-24
This conftest ensures that logging is configured before running tests
"""

import logging
import os
import pytest

# Set the logging level for matplotlib to WARNING to suppress DEBUG messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Create logs directory if it doesn't exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Logging configuration
logging.basicConfig(
    filename="./logs/tests_results.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture(autouse=True)
def run_around_tests():
    """
    Fixture to run before and after each test to ensure logging is configured.
    """
    yield
