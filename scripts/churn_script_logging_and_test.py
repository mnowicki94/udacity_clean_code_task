"""
unit tests for churn library
"""

import logging
import pytest
from scripts.churn_library import ChurnPrediction
from run.constants import DATA_PATH, CAT_COLUMNS, KEEP_COLUMNS


@pytest.fixture(scope="module")
def churn_prediction_fixture():
    """Fixture to create a ChurnPrediction instance and perform initial setup."""
    cp = ChurnPrediction(data_path=DATA_PATH)
    cp.import_data()
    cp.calculate_churn()
    return cp


class TestChurnPrediction:
    """Test suite for ChurnPrediction class methods."""

    @pytest.fixture(autouse=True)
    def setup(self, churn_prediction_fixture):
        """Automatically use the churn_prediction fixture for all tests."""
        self.churn_prediction_instance = churn_prediction_fixture  # Rename the variable

    def test_import_data(self):
        """Test the import_data method."""
        try:
            assert self.churn_prediction_instance.df is not None
            assert not self.churn_prediction_instance.df.empty
            logging.info("Data imported successfully and is not empty.")
        except FileNotFoundError as e:
            logging.error("Error importing data: %s", e)
            pytest.fail(f"Test failed due to error: {e}")

    def test_encoder_helper(self):
        """Test the encoder_helper method for all categorical columns."""
        response = "Churn"
        try:
            self.churn_prediction_instance.df = (
                self.churn_prediction_instance.encoder_helper(
                    self.churn_prediction_instance.df, CAT_COLUMNS, response
                )
            )

            # Iterate through all categorical columns and check the new columns
            for column in CAT_COLUMNS:
                new_column_name = f"{column}_{response}"
                assert new_column_name in self.churn_prediction_instance.df.columns
                assert (
                    not self.churn_prediction_instance.df[new_column_name]
                    .isnull()
                    .any()
                ), f"Column {new_column_name} contains null values."

            logging.info(
                "Encoder helper ran successfully and added new columns for all categories."
            )
        except ValueError as e:
            logging.error("Error during encoding: %s", e)
            pytest.fail(f"Test failed due to error: {e}")

    def test_perform_feature_engineering(self):
        """Test the perform_feature_engineering method."""
        keep_cols = KEEP_COLUMNS
        try:
            x, x_train, x_test, y_train, y_test = (
                self.churn_prediction_instance.perform_feature_engineering(
                    self.churn_prediction_instance.df, keep_cols
                )
            )

            assert x.shape[0] > 0
            assert x_train.shape[0] > 0
            assert x_test.shape[0] > 0
            assert y_train.shape[0] > 0
            assert y_test.shape[0] > 0

            logging.info("Feature engineering completed successfully.")
        except ValueError as e:
            logging.error("Error during feature engineering: %s", e)
            pytest.fail(f"Test failed due to error: {e}")

    def test_train_models(self):
        """Test the train_models method."""
        keep_cols = KEEP_COLUMNS

        # Using a subset of the data to speed up the test
        df_subset = self.churn_prediction_instance.df.head(50)

        x, x_train, x_test, y_train, y_test = (
            self.churn_prediction_instance.perform_feature_engineering(
                df_subset, keep_cols
            )
        )

        try:
            assert x is not None
            assert y_test is not None
            preds = self.churn_prediction_instance.train_models(
                x_train, x_test, y_train
            )
            assert len(preds) == 4  # Should return 4 files
            logging.info(
                "Models trained successfully and predictions generated.")
        except FileNotFoundError as e:
            logging.error(
                "File not found, run modeling first: %s",
                e,
            )
            pytest.fail(f"Test failed due to error: {e}")

    def test_if_error_logging_works(self):
        """Test if error logging works, not actual test of churn functions"""
        if self.churn_prediction_instance.df is not None:
            logging.error("This is a fake error condition for testing.")

        # This assertion is kept as a placeholder to indicate successful
        # completion
        assert True
