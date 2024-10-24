"""
Author: Maciej N.
Date Created: 2024-10-24

This module tests different functions from the churn_library module.
It logs any information or errors into a separate file, and is designed
to ensure that all key functionalities of the ChurnPrediction class are
working correctly.
"""

import logging
import pytest
from scripts.churn_library import ChurnPrediction
from run.constants import DATA_PATH, CAT_COLUMNS, KEEP_COLUMNS


@pytest.fixture(scope="module")
def churn_prediction_fixture():
    """
    Fixture to create a ChurnPrediction instance and perform initial setup.

    Purpose:
        Initialize the ChurnPrediction instance, import data, and calculate churn.
    
    Inputs:
        None (retrieves the path and settings from constants).
    
    Outputs:
        cp (ChurnPrediction): Instance of ChurnPrediction with imported data.
    """
    cp = ChurnPrediction(data_path=DATA_PATH)
    cp.import_data()
    cp.calculate_churn()
    return cp


class TestChurnPrediction:
    """
    Test suite for ChurnPrediction class methods.

    This class runs various tests on key methods of the ChurnPrediction
    class to ensure functionality, including importing data, encoding 
    categorical columns, feature engineering, and model training.

    Author: Maciej N.
    Date Created: 2024-10-24
    """

    @pytest.fixture(autouse=True)
    def setup(self, churn_prediction_fixture):
        """
        Automatically use the churn_prediction fixture for all tests.

        Purpose:
            To initialize the test class with an instance of ChurnPrediction.
        
        Inputs:
            churn_prediction_fixture (ChurnPrediction): Instance created by the fixture.
        
        Outputs:
            None.
        """
        self.churn_prediction_instance = churn_prediction_fixture

    def test_import_data(self):
        """
        Test the import_data method.

        Purpose:
            Ensure that the data is imported successfully and that the DataFrame
            is not empty.
        
        Inputs:
            None.
        
        Outputs:
            None. (Assertions are used to verify the conditions)
        
        Logging:
            Logs success if the data is imported properly, otherwise logs the error.
        """
        try:
            assert self.churn_prediction_instance.df is not None
            assert not self.churn_prediction_instance.df.empty
            logging.info("Data imported successfully and is not empty.")
        except FileNotFoundError as e:
            logging.error("Error importing data: %s", e)
            pytest.fail(f"Test failed due to error: {e}")

    def test_encoder_helper(self):
        """
        Test the encoder_helper method for all categorical columns.

        Purpose:
            To verify that the encoder_helper method successfully encodes all
            categorical columns and adds new columns without null values.
        
        Inputs:
            None. (Uses instance data and predefined categorical columns from constants)
        
        Outputs:
            None. (Assertions are used to verify the conditions)
        
        Logging:
            Logs success if encoding is done correctly, otherwise logs the error.
        """
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
        """
        Test the perform_feature_engineering method.

        Purpose:
            To ensure that the feature engineering method splits the data
            correctly into features and target sets, and provides train/test splits.
        
        Inputs:
            None. (Uses instance data and predefined keep columns)
        
        Outputs:
            None. (Assertions are used to verify the conditions)
        
        Logging:
            Logs success if the feature engineering is done correctly, otherwise logs the error.
        """
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
        """
        Test the train_models method.

        Purpose:
            To verify that the model training method successfully trains models
            and generates predictions based on a small subset of data.
        
        Inputs:
            None. (Uses a subset of the instance data)
        
        Outputs:
            None. (Assertions are used to verify the conditions)
        
        Logging:
            Logs success if models are trained correctly, otherwise logs the error.
        """
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
                "Models trained successfully and predictions generated."
            )
        except FileNotFoundError as e:
            logging.error(
                "File not found, run modeling first: %s",
                e,
            )
            pytest.fail(f"Test failed due to error: {e}")

    def test_if_error_logging_works(self):
        """
        Test if error logging works.

        Purpose:
            This method is used to check if error logging is functioning correctly.
            It does not test actual churn functions but logs a dummy error condition.

        Inputs:
            None.
        
        Outputs:
            None. (The test checks if the logging mechanism works)
        
        Logging:
            Logs a fake error for testing purposes.
        """
        if self.churn_prediction_instance.df is not None:
            logging.error("This is a fake error condition for testing.")

        # This assertion is kept as a placeholder to indicate successful completion
        assert True
