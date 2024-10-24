"""
run script from churn library and constants.py with the following command:
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
    start_time = time.time()

    churn_predictor = ChurnPrediction(DATA_PATH)

    # check if folders exists
    churn_predictor.check_folder_exists("./images/eda")
    churn_predictor.check_folder_exists("./images/results")
    churn_predictor.check_folder_exists("./models")

    # import data
    logging.info("Importing data...")
    churn_predictor.import_data()
    logging.info("Importing data success")

    # calculate churn
    logging.info("Calculating churn...")
    churn_predictor.df = churn_predictor.calculate_churn()
    logging.info("Churn calculated")

    # perform eda
    logging.info("Performing EDA...")
    churn_predictor.perform_eda(churn_predictor.df, QUANT_COLUMNS)
    logging.info("Performing EDA success")

    # perform encoding
    logging.info("Performing encoding...")
    churn_predictor.df = churn_predictor.encoder_helper(
        churn_predictor.df,
        CAT_COLUMNS,
        "Churn",
    )
    logging.info("Performing encoding success")

    # perform feature engineering
    logging.info("Performing feature engineering...")
    (x, x_train, x_test, y_train, y_test) = churn_predictor.perform_feature_engineering(
        churn_predictor.df, KEEP_COLUMNS
    )
    logging.info("Performing feature engineering success")

    # train models
    logging.info("Performing model training...")

    (y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr) = (
        churn_predictor.train_models(x_train, x_test, y_train)
    )
    logging.info("Model successfully trained")

    # prepare dictionaries with actual and predicted data
    actual_data = {"y_train": y_train, "y_test": y_test}
    predicted_data = {
        "y_train_preds_rf": y_train_preds_rf,
        "y_train_preds_lr": y_train_preds_lr,
        "y_test_preds_rf": y_test_preds_rf,
        "y_test_preds_lr": y_test_preds_lr,
    }

    # plot results
    logging.info("Performing classification report...")
    churn_predictor.classification_report_image(actual_data, predicted_data)
    logging.info("classification report ready")

    # feature importance plot
    logging.info("Performing feature importance plot...")
    churn_predictor.feature_importance_plot(
        "./models/rfc_model.pkl", x, "./images/results/importances_plot.png"
    )
    logging.info("feature importance plot ready")

    # roc plot
    logging.info("Performing ROC plot...")
    churn_predictor.roc_plot(
        "./models/rfc_model.pkl",
        "./models/lrc_model.pkl",
        x_test,
        y_test,
        "./images/results/roc.png",
    )
    logging.info("ROC plot ready")

    # random forest explainer plot
    logging.info("Performing RF explainer plot...")
    churn_predictor.rf_explainer_plot(
        "./models/rfc_model.pkl", x_test, "./images/results/rf_explainer.png"
    )
    logging.info("RF explainer plot ready")

    logging.info(
        "Finished, it took: %s minutes", (time.time() - start_time) / 60
    )  # This is correct
