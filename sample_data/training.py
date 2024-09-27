import logging
import os
import random
import shutil
!pip install pyspark
import pandas as pd
#Load the librarys
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn

import requests
from bs4 import BeautifulSoup

# Load the libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, countDistinct, lit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import sys
import os

# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import pickle
!pip install joblib
from joblib import Parallel, delayed 
import joblib 
from sklearn.preprocessing import OneHotEncoder


def create_spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_dataset(spark, file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

def rename_columns(df):
    return df.select([col(c).alias(c.lower()) for c in df.columns])

def rename_specific_columns(df):
    df = df.withColumnRenamed("saving accounts", "savings_account")
    df = df.withColumnRenamed("checking account", "checking_account")
    df = df.withColumnRenamed("credit amount", "credit_amount")
    return df

def get_most_frequent_category(df, column_name):
    return df.select(column_name).groupBy(column_name).agg(count("*").alias("count")).orderBy(col("count").desc()).collect()[0][column_name]

def fill_missing_values(df, column_name, most_frequent_value):
    return df.withColumn(column_name, when(col(column_name) == "NA", most_frequent_value).otherwise(col(column_name)))

def create_new_features(df):
    df = df.withColumn("credit_amount_range", when(col("credit_amount") < 5000, "low")
                                                .when(col("credit_amount").between(5000, 10000), "medium")
                                                .otherwise("high"))
    df = df.withColumn("age_group", when(col("age") < 30, "young")
                                      .when(col("age").between(30, 60), "middle-aged")
                                      .otherwise("senior"))
    return df

def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    conf = confusion_matrix(actual, pred)
    betaf = fbeta_score(actual, pred, beta=1.0)  # Specify beta value
    return acc, conf, betaf

def one_hot_encode_columns(df, columns, drop_first=True):
    for column in columns:
        df = df.merge(pd.get_dummies(df[column], drop_first=drop_first, prefix=column), left_index=True, right_index=True)
    df.drop(columns=columns, inplace=True)
    return df

def preprocess_data(df):
    df = df.toPandas()
    df = df.drop('_c0', axis=1)
    
    # Columns to be one-hot encoded
    categorical_columns = ["purpose", "sex", "housing", "savings_account", "checking_account", "age_group", "credit_amount_range"]
    
    # Apply one-hot encoding
    df = one_hot_encode_columns(df, categorical_columns)
    
    # Encode the target variable
    df['risk'].replace({'bad': 1, 'good': 0}, inplace=True)
    
    return df


# Example usage within an Airflow task
def process_credit_data():
    spark = create_spark_session("CreditDataProcessing")
    df_credit = load_dataset(spark, "german_credit_data.csv")
    df_credit = rename_columns(df_credit)
    df_credit = rename_specific_columns(df_credit)
    most_freq_savings = get_most_frequent_category(df_credit, "savings_account")
    most_freq_checking = get_most_frequent_category(df_credit, "checking_account")
    df_credit = fill_missing_values(df_credit, "savings_account", most_freq_savings)
    df_credit = fill_missing_values(df_credit, "checking_account", most_freq_checking)
    df_credit = create_new_features(df_credit)
    # Additional processing can be added here
    df_credit=preprocess_data(df_credit)
    return df_credit

df=process_credit_data()
df.columns = df.columns.astype(str)

X=df.drop('risk', axis=1)
y=df['risk']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model= RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc, conf, betaf = eval_metrics(y_test, y_pred)
print("Accuracy",{acc})
# Save the model as a pickle in a file 
joblib.dump(model, 'random_forest.pkl') 