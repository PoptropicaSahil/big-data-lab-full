import pandas as pd

# Load the librarys
# Load the libraries
from pyspark.sql import SparkSession

# Load the librarys
# Load the libraries
import joblib
from pyspark.sql.functions import col, count, when
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from sklearn.preprocessing import OneHotEncoder

import os
# Get the current working directory
cwd = os.getcwd()
# Load the librarys


# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from xgboost import XGBClassifier

import tempfile
from io import StringIO


class TrainingUtils:
    def __init__(self):
        pass

    def create_spark_session(self, app_name):
        return SparkSession.builder.appName(app_name).getOrCreate()  # type: ignore

    def load_dataset(self, spark, file_path):
        # df = pd.read_csv(file_path, index_col=0)
        # df_spark = spark.createDataFrame(df)
        # return df_spark

        # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        #     temp_file.write(file_path.file.read())
        #     file_path = temp_file.name

        #     return spark.read.csv(file_path, header=True, inferSchema=True)
        # file_stream = io.BytesIO(file_path.file.read())

        return spark.read.csv(
            StringIO(str(file_path.file.read(), "utf-16")),
            header=True,
            inferSchema=True,
        )

    def rename_columns(self, df):
        return df.select([col(c).alias(c.lower()) for c in df.columns])

    def rename_specific_columns(self, df):
        df = df.withColumnRenamed("saving accounts", "savings_account")
        df = df.withColumnRenamed("checking account", "checking_account")
        df = df.withColumnRenamed("credit amount", "credit_amount")
        return df

    def get_most_frequent_category(self, df, column_name):
        return (
            df.select(column_name)
            .groupBy(column_name)
            .agg(count("*").alias("count"))
            .orderBy(col("count").desc())
            .collect()[0][column_name]
        )

    def fill_missing_values(self, df, column_name, most_frequent_value):
        return df.withColumn(
            column_name,
            when(col(column_name) == "NA", most_frequent_value).otherwise(
                col(column_name)
            ),
        )

    def create_new_features(self, df):
        df = df.withColumn(
            "credit_amount_range",
            when(col("credit_amount") < 5000, "low")
            .when(col("credit_amount").between(5000, 10000), "medium")
            .otherwise("high"),
        )
        df = df.withColumn(
            "age_group",
            when(col("age") < 30, "young")
            .when(col("age").between(30, 60), "middle-aged")
            .otherwise("senior"),
        )
        return df

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        conf = confusion_matrix(actual, pred)
        betaf = fbeta_score(actual, pred, beta=1.0)  # Specify beta value
        return acc, conf, betaf


    def preprocess_data_training(self, df):
        df = df.toPandas()

        try:
            df = df.drop('_c0', axis=1)
        except KeyError:
            pass

        # Define the columns to be one-hot encoded
        columns_to_encode = ['purpose', 'sex', 'housing', 'savings_account', 'checking_account', 'age_group','credit_amount_range']


        ohe = OneHotEncoder(drop='first', sparse=False)

        #One-hot-encode the categorical columns.
        #Unfortunately outputs an array instead of dataframe.
        array_hot_encoded = ohe.fit_transform(df[columns_to_encode])
        #Convert it to df
        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df.index)
        #Extract only the columns that didnt need to be encoded
        data_other_cols = df.drop(columns=columns_to_encode)

        #Concatenate the two dataframes :
        df = pd.concat([data_hot_encoded, data_other_cols], axis=1)

        joblib.dump(ohe, "ohe.joblib")

        # # Apply one-hot encoding
        # df = one_hot_encode_columns(df, categorical_columns)

        # Encode the target variable
        df['risk'].replace({'bad': 1, 'good': 0}, inplace=True)
        
        return df

    def preprocess_data_inference(self, df):
        df = df.toPandas()
        # df = df.drop('_c0', axis=1)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)  

        print(df.head())
        print(df.tail())

        print(f"df in step will be {df} cols are {df.columns}, shape = {df.shape}")
        try:
            df = df.drop('_c0', axis=1)
        except KeyError:
            pass

        # Define the columns to be one-hot encoded
        columns_to_encode = ['purpose', 'sex', 'housing', 'savings_account', 'checking_account', 'age_group','credit_amount_range']

        # ohe=joblib.load('ohe.joblib')

        # Load the joblib file using the full path
        ohe = joblib.load(os.path.join(cwd, 'utils/ohe.joblib'))



        #One-hot-encode the categorical columns.
        #Unfortunately outputs an array instead of dataframe.
        array_hot_encoded = ohe.transform(df[columns_to_encode])
        #Convert it to df
        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df.index)
        #Extract only the columns that didnt need to be encoded
        data_other_cols = df.drop(columns=columns_to_encode)

        #Concatenate the two dataframes :
        df = pd.concat([data_hot_encoded, data_other_cols], axis=1)
        print(df.shape)

        # Encode the target variable
        #df['risk'].replace({'bad': 1, 'good': 0}, inplace=True)

        return df

        