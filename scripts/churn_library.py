"""
Author: Maciej N. 
Date Created: 2024-10-24

This modules stores functions needed to perform churn modeling.
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


os.environ["QT_QPA_PLATFORM"] = "offscreen"


class ChurnPrediction:
    """
    Churn Prediction class encapsulates all the methods for EDA, feature engineering,
    model training, and evaluation related to predicting customer churn.
    """

    def __init__(self, data_path=None):
        """
        Initialize the class with optional data path.
        """
        self.data_path = data_path
        self.df = None
        self.models = {}

    @staticmethod
    def check_folder_exists(folder_path):
        """
        check if folder exists
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def import_data(self):
        """
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        """
        self.df = pd.read_csv(self.data_path)

    def calculate_churn(self):
        """
        returns a dataframe with a new column 'Churn'

        input:
                df: pandas dataframe
        output:
                df: pandas dataframe with new column 'Churn'
        """

        self.df["Churn"] = self.df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        return self.df

    @staticmethod
    def save_plot(plot_func, filename, *args, **kwargs):
        """
        Helper function to create and save a plot.

        Parameters:
            plot_func: The plotting function to call (e.g., plt.hist, sns.histplot).
            filename: The path where the plot will be saved.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        plt.figure(figsize=(20, 10))
        plot_func(*args, **kwargs)
        plt.savefig(filename)
        plt.close()

    def perform_eda(self, df, quant_columns):
        """
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe
                quant_columns: list of numerical columns

        output:
                None
        """

        # Plot the churn histogram
        self.save_plot(df["Churn"].hist, "./images/eda/churn.png")

        # Plot the customer age histogram
        self.save_plot(
            df["Customer_Age"].hist,
            "./images/eda/customer_age.png")

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
        sns.heatmap(
            df[quant_columns].corr(),
            annot=False,
            cmap="Dark2_r",
            linewidths=2)
        plt.savefig("./images/eda/correlations.png")
        plt.close()

    @staticmethod
    def encoder_helper(df, category_lst, response):
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming
                variables or index y column]

        output:
                df: pandas dataframe with new columns for
        """

        for column in category_lst:
            churn_groups = df.groupby(column)[response].mean()

            df[f"{column}_{response}"] = df[column].map(churn_groups)

        return df

    @staticmethod
    def perform_feature_engineering(df, keep_cols):
        """
        input:
                df: pandas dataframe
                variables or index y column]

        output:
                x: x data
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        """
        y = df["Churn"]
        x = pd.DataFrame()

        x[keep_cols] = df[keep_cols]

        # This cell may take up to 15-20 minutes to run
        # train test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
        )

        return (
            x,
            x_train,
            x_test,
            y_train,
            y_test,
        )

    def train_models(self, x_train, x_test, y_train):
        """
        train, store model results: images + scores, and store models
        input:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        """

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        # save best model
        joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
        joblib.dump(lrc, "./models/lrc_model.pkl")

        return (
            y_train_preds_rf,
            y_train_preds_lr,
            y_test_preds_rf,
            y_test_preds_lr)

    @staticmethod
    def classification_report_image(actual_data, predicted_data):
        """
        Produces classification report for training and testing results and stores report as image
        in images folder.

        input:
            actual_data: dictionary with 'y_train' and 'y_test' actual values
            predicted_data: dictionary with predicted values for each model (random forest
            and logistic regression)

        output:
            None
        """

        # Unpack actual and predicted values
        y_train = actual_data["y_train"]
        y_test = actual_data["y_test"]

        y_train_preds_rf = predicted_data["y_train_preds_rf"]
        y_train_preds_lr = predicted_data["y_train_preds_lr"]
        y_test_preds_rf = predicted_data["y_test_preds_rf"]
        y_test_preds_lr = predicted_data["y_test_preds_lr"]

        # plot
        plt.rc("figure", figsize=(7, 7))
        plt.text(
            0.01,
            1.25,
            str("Random Forest Train"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_test, y_test_preds_rf)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.6,
            str("Random Forest Test"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_train, y_train_preds_rf)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.axis("off")
        plt.savefig("./images/results/rf_results.png")
        plt.close()

        plt.rc("figure", figsize=(7, 7))
        plt.text(
            0.01,
            1.25,
            str("Logistic Regression Train"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_train, y_train_preds_lr)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.6,
            str("Logistic Regression Test"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_test, y_test_preds_lr)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.axis("off")
        plt.savefig("./images/results/lr_results.png")
        plt.close()

    @staticmethod
    def feature_importance_plot(rf_model, x_data, output_pth):
        """
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                x_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        """
        # import model
        rfc_model = joblib.load(rf_model)

        # Calculate feature importances
        importances = rfc_model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        plt.title("Feature Importance")
        plt.ylabel("Importance")

        plt.bar(range(x_data.shape[1]), importances[indices])

        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        plt.close()

    @staticmethod
    def roc_plot(rf_model, lr_model, x_test, y_test, output_pth):
        """
        creates and stores roc plot in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        """
        rfc_model = joblib.load(rf_model)
        lrc_model = joblib.load(lr_model)

        plt.figure(figsize=(15, 8))
        ax = plt.gca()

        RocCurveDisplay.from_estimator(
            rfc_model, x_test, y_test, ax=ax, alpha=0.8)
        RocCurveDisplay.from_estimator(
            lrc_model, x_test, y_test, ax=ax, alpha=0.8)

        plt.savefig(output_pth)
        plt.close()

    @staticmethod
    def rf_explainer_plot(rf_model, x_test, output_pth):
        """
        creates and stores explainer of random forest plot in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        """
        # Load the model
        rfc_model = joblib.load(rf_model)

        # Create the SHAP explainer
        explainer = shap.TreeExplainer(rfc_model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(x_test)

        # Create the summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)

        # Save the plot as a PNG file
        plt.savefig(output_pth, format="png", dpi=300)
        plt.close()
