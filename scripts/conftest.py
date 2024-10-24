"""
Author: Maciej N. 
Date Created: 2024-10-24

This conftest module ensures that logging is configured before running tests in the churn prediction library.
It creates a logs directory (if not present) and configures the logging settings for the test results.
"""

import logging
import os
import pytest

# Set the logging level for matplotlib to WARNING to suppress unnecessary DEBUG messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Create the logs directory if it doesn't already exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Configure logging to write test results to the specified log file
logging.basicConfig(
    filename="./logs/tests_results.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

@pytest.fixture(autouse=True)
def run_around_tests():
    """
    Pytest fixture that runs before and after each test to ensure logging is properly configured.
    This fixture is automatically applied to all tests.
    """
    yield
