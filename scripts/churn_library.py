"""
Author: Maciej N.
Date Created: 2024-10-24

This module stores functions and methods needed to perform churn modeling,
including data loading, exploratory data analysis (EDA), feature engineering,
and model training.
"""

# import libraries
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report


# Set an environment variable to prevent display-related errors when running on servers
os.environ["QT_QPA_PLATFORM"] = "offscreen"


class ChurnPrediction:
    """
    Churn Prediction class encapsulates all the methods for EDA, feature engineering,
    model training, and evaluation related to predicting customer churn.
    """

    def __init__(self, data_path=None):
        """
        Initialize the class with an optional data path to load the dataset.

        Input:
        - data_path: str, default None. Path to the CSV file containing customer data.
        """
        self.data_path = data_path
        self.df = None
        self.models = {}

    @staticmethod
    def check_folder_exists(folder_path):
        """
        Check if a folder exists; if not, create it.

        Input:
        - folder_path: str. The path of the folder to check or create.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def import_data(self):
        """
        Load data from the provided CSV file path into a pandas DataFrame.

        Output:
        - self.df: pandas DataFrame containing the loaded data.
        """
        self.df = pd.read_csv(self.data_path)

    def calculate_churn(self):
        """
        Add a 'Churn' column to the DataFrame based on the 'Attrition_Flag' column.
        A value of 1 is assigned if 'Attrition_Flag' indicates 'Attrited Customer', otherwise 0.

        Output:
        - self.df: pandas DataFrame with a new 'Churn' column.
        """
        self.df["Churn"] = self.df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        return self.df

    @staticmethod
    def save_plot(plot_func, filename, *args, **kwargs):
        """
        Helper function to create and save a plot.

        Input:
        - plot_func: Callable. The plotting function to call (e.g., plt.hist, sns.histplot).
        - filename: str. The path where the plot will be saved.
        - *args: Additional positional arguments for the plotting function.
        - **kwargs: Additional keyword arguments for the plotting function.
        """
        plt.figure(figsize=(20, 10))
        plot_func(*args, **kwargs)
        plt.savefig(filename)
        plt.close()

    def perform_eda(self, df, quant_columns):
        """
        Perform exploratory data analysis (EDA) on the DataFrame and save figures to the 'images' folder.

        Input:
        - df: pandas DataFrame. The dataset to analyze.
        - quant_columns: list. List of numerical columns for generating correlation heatmap.
        """
        # Plot the churn histogram
        self.save_plot(df["Churn"].hist, "./images/eda/churn.png")

        # Plot the customer age histogram
        self.save_plot(df["Customer_Age"].hist, "./images/eda/customer_age.png")

        # Plot the marital status bar chart
        self.save_plot(
            df.Marital_Status.value_counts("normalize").plot,
            "./images/eda/marital_status.png",
            kind="bar",
        )

        # Plot the total transaction count density plot
        self.save_plot(
            sns.histplot,
            "./images/eda/total_trans_ct.png",
            df["Total_Trans_Ct"],
            stat="density",
            kde=True,
        )

        # Plot the correlation heatmap
        plt.figure(figsize=(20, 10))  # Heatmap has a different setup
        sns.heatmap(df[quant_columns].corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("./images/eda/correlations.png")
        plt.close()

    @staticmethod
    def encoder_helper(df, category_lst, response):
        """
        Helper function to encode categorical columns by calculating the proportion
        of churn for each category in the specified column.

        Input:
        - df: pandas DataFrame. The dataset containing categorical columns.
        - category_lst: list. List of categorical column names to encode.
        - response: str. The target column name ('Churn') used for encoding.

        Output:
        - df: pandas DataFrame with new encoded columns.
        """
        for column in category_lst:
            churn_groups = df.groupby(column)[response].mean()
            df[f"{column}_{response}"] = df[column].map(churn_groups)

        return df

    @staticmethod
    def perform_feature_engineering(df, keep_cols):
        """
        Split the DataFrame into training and test datasets based on selected columns.

        Input:
        - df: pandas DataFrame. The dataset to split.
        - keep_cols: list. List of column names to keep for the model.

        Outputs:
        - x: pandas DataFrame containing selected feature columns.
        - x_train: X training data.
        - x_test: X testing data.
        - y_train: y training data.
        - y_test: y testing data.
        """
        y = df["Churn"]
        x = pd.DataFrame()
        x[keep_cols] = df[keep_cols]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
        )

        return x, x_train, x_test, y_train, y_test

    def train_models(self, x_train, x_test, y_train):
        """
        Train RandomForest and LogisticRegression models using the training data.
        Save the trained models and return predicted values for both models.

        Input:
        - x_train: pandas DataFrame. The training feature data.
        - x_test: pandas DataFrame. The testing feature data.
        - y_train: pandas Series. The target training data.

        Outputs:
        - y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr: Predicted values for both models.
        """
        # Random Forest and Logistic Regression model setup
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        # Grid search for Random Forest
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)
        lrc.fit(x_train, y_train)

        # Get predictions
        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        # Save models
        joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
        joblib.dump(lrc, "./models/lrc_model.pkl")

        return y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr

    @staticmethod
    def classification_report_image(actual_data, predicted_data):
        """
        Generate and save classification reports for both training and testing sets.

        Input:
        - actual_data: dict. Contains actual 'y_train' and 'y_test' values.
        - predicted_data: dict. Contains predicted values for RandomForest and LogisticRegression models.
        """
        y_train = actual_data["y_train"]
        y_test = actual_data["y_test"]

        y_train_preds_rf = predicted_data["y_train_preds_rf"]
        y_train_preds_lr = predicted_data["y_train_preds_lr"]
        y_test_preds_rf = predicted_data["y_test_preds_rf"]
        y_test_preds_lr = predicted_data["y_test_preds_lr"]

        # Plot and save RandomForest results
        plt.rc("figure", figsize=(7, 7))
        plt.text(
            0.01,
            1.25,
            "Random Forest Train",
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_test, y_test_preds_rf)),
            {"fontsize": 10},
        )
        plt.text(0.01, 0.6, "Random Forest Test", {"fontsize": 10})
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_train, y_train_preds_rf)),
            {"fontsize": 10},
        )
        plt.axis("off")
        plt.savefig("./images/results/rf_results.png")
        plt.close()

        # Plot and save LogisticRegression results
        plt.rc("figure", figsize=(7, 7))
        plt.text(
            0.01,
            1.25,
            "Logistic Regression Train",
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_train, y_train_preds_lr)),
            {"fontsize": 10},
        )
        plt.text(0.01, 0.6, "Logistic Regression Test", {"fontsize": 10})
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_test, y_test_preds_lr)),
            {"fontsize": 10},
        )
        plt.axis("off")
        plt.savefig("./images/results/lr_results.png")
        plt.close()

    @staticmethod
    def feature_importance_plot(rf_model, x_data, output_pth):
        """
        Generate and save a bar plot showing feature importance based on the trained RandomForest model.

        Input:
        - rf_model: str. Path to the saved RandomForest model file.
        - x_data: pandas DataFrame. Feature data used for training.
        - output_pth: str. Path to save the plot.
        """
        rfc_model = joblib.load(rf_model)
        importances = rfc_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.bar(range(x_data.shape[1]), importances[indices])
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        plt.close()

    @staticmethod
    def roc_plot(rf_model, lr_model, x_test, y_test, output_pth):
        """
        Generate and save ROC curve plots for RandomForest and LogisticRegression models.

        Input:
        - rf_model: str. Path to the saved RandomForest model file.
        - lr_model: str. Path to the saved LogisticRegression model file.
        - x_test: pandas DataFrame. Test feature data.
        - y_test: pandas Series. True test labels.
        - output_pth: str. Path to save the plot.
        """
        rfc_model = joblib.load(rf_model)
        lrc_model = joblib.load(lr_model)

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        RocCurveDisplay.from_estimator(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
        RocCurveDisplay.from_estimator(lrc_model, x_test, y_test, ax=ax, alpha=0.8)
        plt.savefig(output_pth)
        plt.close()

    @staticmethod
    def rf_explainer_plot(rf_model, x_test, output_pth):
        """
        Generate and save a SHAP explainer plot for the RandomForest model.

        Input:
        - rf_model: str. Path to the saved RandomForest model file.
        - x_test: pandas DataFrame. Test feature data.
        - output_pth: str. Path to save the plot.
        """
        rfc_model = joblib.load(rf_model)
        explainer = shap.TreeExplainer(rfc_model)
        shap_values = explainer.shap_values(x_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
        plt.savefig(output_pth, format="png", dpi=300)
        plt.close()
