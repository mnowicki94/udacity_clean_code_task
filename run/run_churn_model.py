"""
Author: Maciej N.
Date Created: 2024-10-24

This script executes the ChurnPrediction class methods from the churn_library module
and uses constants from the constants.py file. It performs the full process from
data import, churn calculation, EDA, feature engineering, model training, and result
visualization. The script logs the progress and outcomes in a separate log file.

To run this script, use the following command:
    make run-modeling
"""

import os
import logging
import time
from scripts.churn_library import ChurnPrediction
from run.constants import DATA_PATH, QUANT_COLUMNS, CAT_COLUMNS, KEEP_COLUMNS

# Create logs directory if it doesn't exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")

logging.basicConfig(
    filename="./logs/modeling_results.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    """
    Main script to run the churn prediction model training and reporting.

    Purpose:
        This script automates the churn prediction pipeline, including:
        - Data importing and preprocessing.
        - Exploratory Data Analysis (EDA).
        - Encoding categorical variables.
        - Feature engineering.
        - Model training (Random Forest and Logistic Regression).
        - Generating and saving model evaluation reports and plots.
    
    Inputs:
        The script uses constants from the `constants.py` file:
        - DATA_PATH (str): Path to the dataset.
        - QUANT_COLUMNS (list): List of numerical columns for EDA.
        - CAT_COLUMNS (list): List of categorical columns for encoding.
        - KEEP_COLUMNS (list): List of features to retain for model training.
    
    Outputs:
        Various outputs are logged or saved to files, including:
        - Logs for each step in `./logs/modeling_results.log`.
        - EDA plots, classification reports, and model performance plots.
        - Trained model files stored in the `./models` directory.
    
    Logging:
        Logs are written to the `./logs/modeling_results.log` file to track progress 
        and capture any errors during execution.
    """

    start_time = time.time()

    churn_predictor = ChurnPrediction(DATA_PATH)

    # Check if necessary folders exist, create if they don't
    churn_predictor.check_folder_exists("./images/eda")
    churn_predictor.check_folder_exists("./images/results")
    churn_predictor.check_folder_exists("./models")

    # Import data
    logging.info("Importing data...")
    churn_predictor.import_data()
    logging.info("Data imported successfully.")

    # Calculate churn
    logging.info("Calculating churn...")
    churn_predictor.df = churn_predictor.calculate_churn()
    logging.info("Churn calculated successfully.")

    # Perform Exploratory Data Analysis (EDA)
    logging.info("Performing EDA...")
    churn_predictor.perform_eda(churn_predictor.df, QUANT_COLUMNS)
    logging.info("EDA completed successfully.")

    # Perform encoding on categorical columns
    logging.info("Encoding categorical columns...")
    churn_predictor.df = churn_predictor.encoder_helper(
        churn_predictor.df,
        CAT_COLUMNS,
        "Churn",
    )
    logging.info("Encoding completed successfully.")

    # Perform feature engineering
    logging.info("Performing feature engineering...")
    (x, x_train, x_test, y_train, y_test) = churn_predictor.perform_feature_engineering(
        churn_predictor.df, KEEP_COLUMNS
    )
    logging.info("Feature engineering completed successfully.")

    # Train models (Random Forest and Logistic Regression)
    logging.info("Training models...")
    (y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr) = (
        churn_predictor.train_models(x_train, x_test, y_train)
    )
    logging.info("Models trained successfully.")

    # Prepare dictionaries with actual and predicted data for model evaluation
    actual_data = {"y_train": y_train, "y_test": y_test}
    predicted_data = {
        "y_train_preds_rf": y_train_preds_rf,
        "y_train_preds_lr": y_train_preds_lr,
        "y_test_preds_rf": y_test_preds_rf,
        "y_test_preds_lr": y_test_preds_lr,
    }

    # Generate classification report image
    logging.info("Generating classification report...")
    churn_predictor.classification_report_image(actual_data, predicted_data)
    logging.info("Classification report generated successfully.")

    # Generate feature importance plot
    logging.info("Generating feature importance plot...")
    churn_predictor.feature_importance_plot(
        "./models/rfc_model.pkl", x, "./images/results/importances_plot.png"
    )
    logging.info("Feature importance plot generated successfully.")

    # Generate ROC plot
    logging.info("Generating ROC plot...")
    churn_predictor.roc_plot(
        "./models/rfc_model.pkl",
        "./models/lrc_model.pkl",
        x_test,
        y_test,
        "./images/results/roc.png",
    )
    logging.info("ROC plot generated successfully.")

    # Generate Random Forest explainer plot
    logging.info("Generating Random Forest explainer plot...")
    churn_predictor.rf_explainer_plot(
        "./models/rfc_model.pkl", x_test, "./images/results/rf_explainer.png"
    )
    logging.info("Random Forest explainer plot generated successfully.")

    # Logging the total time taken for execution
    logging.info(
        "Modeling pipeline finished successfully. It took: %s minutes",
        (time.time() - start_time) / 60,
    )
