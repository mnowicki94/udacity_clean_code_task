"""
Author: Maciej N.
Date Created: 2024-10-24

This module contains constant variables used in the churn prediction library.
These constants include paths to data, lists of quantitative and categorical
columns, and lists of columns to be used for feature engineering in model training.
"""

# Path to the data file containing customer information for churn analysis
DATA_PATH = "./data/bank_data.csv"

# List of quantitative columns used for Exploratory Data Analysis (EDA) and model features
QUANT_COLUMNS = [
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
]

# List of categorical columns used for encoding in the feature engineering process
CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

# List of columns to be retained for model training after feature engineering
KEEP_COLUMNS = [
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
    # Encoded categorical columns for churn prediction
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]
