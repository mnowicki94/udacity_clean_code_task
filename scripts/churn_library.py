# library doc string
"""
Library of functions to find customers who are likely to churn
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


def check_folder_exists(folder_path):
    """
    check if folder exists
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data = pd.read_csv(pth)

    return data


def perform_eda(df, quant_columns):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

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

    # Create the 'Churn' column
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Plot the churn histogram
    save_plot(df["Churn"].hist, "./images/eda/churn.png")

    # Plot the customer age histogram
    save_plot(df["Customer_Age"].hist, "./images/eda/customer_age.png")

    # Plot the marital status bar chart
    save_plot(
        df.Marital_Status.value_counts("normalize").plot,
        "./images/eda/marital_status.png",
        kind="bar",
    )

    # Plot the total transaction count density plot
    save_plot(
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

    # Iterate over each column
    for column in category_lst:
        # Group by the column and calculate the mean of Churn
        churn_groups = df.groupby(column)[response].mean()

        # Use the mapping to create a new column for encoded churn values
        df[f"{column}_{response}"] = df[column].map(churn_groups)

    return df


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


def train_models(x_train, x_test, y_train):
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

    return (y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    plt.rc("figure", figsize=(7, 7))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
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
    )  # approach improved by OP -> monospace!
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
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("./images/results/rf_results.png")
    plt.close()

    ####
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
    )  # approach improved by OP -> monospace!
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
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("./images/results/lr_results.png")
    plt.close()


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

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


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

    RocCurveDisplay.from_estimator(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lrc_model, x_test, y_test, ax=ax, alpha=0.8)

    plt.savefig(output_pth)
    plt.close()


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
    plt.figure(figsize=(10, 6))  # Optionally set the figure size
    shap.summary_plot(
        shap_values, x_test, plot_type="bar", show=False
    )  # Use show=False to avoid displaying it immediately

    # Save the plot as a PNG file
    plt.savefig(
        output_pth, format="png", dpi=300
    )  # Adjust dpi for resolution if needed
    plt.close()  # Close the plot to avoid display


if __name__ == "__main__":
    # check if folder exists
    check_folder_exists("./images/eda")
    check_folder_exists("./images/results")
    check_folder_exists("./models")

    # import data
    df = import_data("./data/bank_data.csv")

    # perform eda
    perform_eda(
        df,
        [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
        ],
    )

    # perform feature engineering
    df = encoder_helper(
        df,
        [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ],
        "Churn",
    )

    # perform feature engineering
    (x, x_train, x_test, y_train, y_test) = perform_feature_engineering(
        df,
        [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn",
        ],
    )

    # train models
    (y_train_preds_rf, y_train_preds_lr, y_test_preds_rf, y_test_preds_lr) = (
        train_models(x_train, x_test, y_train)
    )

    # plot results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # feature importance plot
    feature_importance_plot(
        "./models/rfc_model.pkl", x, "./images/results/importances_plot.png"
    )

    # roc plot
    roc_plot(
        "./models/rfc_model.pkl",
        "./models/lrc_model.pkl",
        x_test,
        y_test,
        "./images/results/roc.png",
    )

    # random forest explainer plot
    rf_explainer_plot(
        "./models/rfc_model.pkl", x_test, "./images/results/rf_explainer.png"
    )

    # TO IMPROVE
    # CLEAN
    # add some comments
    # Re-organize each script to work as a class.
    # Update functions to move constants to their own constants.py file, which can then be passed to the necessary functions, rather than being created as variables within functions.
    # pylint 10/10
