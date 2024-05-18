import os

import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from sklearn.preprocessing import OneHotEncoder

# Get the current working directory
cwd = os.getcwd()


class TrainingUtils:
    def __init__(self):
        pass

    def create_spark_session(self, app_name):
        """Create spark Session"""
        return SparkSession.builder.appName(app_name).getOrCreate()  # type: ignore

    def load_dataset(self, spark, file_path):
        """Load the training file from the path"""
        return spark.read.csv(file_path, header=True, inferSchema=True)

    def data_val_checks(self, df):
        """Check if all columns are present in the data. The flag takes value 1 
        if there is any missing column or a CRITICAL column has missing values""" 
        data_validation_check_flag = 0

        # Check for presence of columns
        cols_in_df = df.columns
        expected_cols = ['_c0', 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                         'Checking account', 'Credit amount', 'Duration', 'Purpose']

        if set(cols_in_df) != set(expected_cols):
            data_validation_check_flag = 1

        # TODO:
        # Add checks if any column except the  'Saving accounts', 'Checking account' have any empty values
        # The has_missing_values column will be True if any of the selected columns
        # expected_filled_cols = ['Age', 'Sex', 'Job', 'Housing', 'Credit amount', 'Duration', 'Purpose']
        # missing_conditions = [col(c).isNull() for c in expected_filled_cols]
        # has_missing_values = any(missing_conditions) # not working

        # if has_missing_values == True:
        #     data_validation_check_flag = 1


        return data_validation_check_flag
    
    def rename_columns(self, df):
        """Rename columns to lowercase for convinience"""
        return df.select([col(c).alias(c.lower()) for c in df.columns])

    def rename_specific_columns(self, df):
        """Rename specific columns with spaces in their names"""
        df = df.withColumnRenamed("saving accounts", "savings_account")
        df = df.withColumnRenamed("checking account", "checking_account")
        df = df.withColumnRenamed("credit amount", "credit_amount")
        return df

    def get_most_frequent_category(self, df, column_name):
        """Get most frequent category in the column - useful while missing value imputing"""
        return (
            df.select(column_name)
            .groupBy(column_name)
            .agg(count("*").alias("count"))
            .orderBy(col("count").desc())
            .collect()[0][column_name]
        )

    def fill_missing_values(self, df, column_name, most_frequent_value):
        """Fill missing values with most frequent values"""
        return df.withColumn(
            column_name,
            when(col(column_name) == "NA", most_frequent_value).otherwise(
                col(column_name)
            ),
        )

    def create_new_features(self, df):
        """Bin few columns to managable values"""
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
        """Generate predictions"""
        acc = accuracy_score(actual, pred)
        conf = confusion_matrix(actual, pred)
        betaf = fbeta_score(actual, pred, beta=1.0)  # Specify beta value
        return acc, conf, betaf

    def preprocess_data_training(self, df):
        """Preprocessing training data"""
        df = df.toPandas()

        # This column comes up wierdly when the File is loaded using spark
        try:
            df = df.drop("_c0", axis=1)
        except KeyError:
            pass

        # Define the columns to be one-hot encoded
        columns_to_encode = [
            "purpose",
            "sex",
            "housing",
            "savings_account",
            "checking_account",
            "age_group",
            "credit_amount_range",
        ]

        ohe = OneHotEncoder(drop="first", sparse=False)

        # One-hot-encode the categorical columns.
        # Unfortunately outputs an array instead of dataframe.
        array_hot_encoded = ohe.fit_transform(df[columns_to_encode])
        # Convert it to df
        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df.index)
        # Extract only the columns that didnt need to be encoded
        data_other_cols = df.drop(columns=columns_to_encode)

        # Concatenate the two dataframes :
        df = pd.concat([data_hot_encoded, data_other_cols], axis=1)

        # Save the OHE
        joblib.dump(ohe, "ohe.joblib")

        # # Apply one-hot encoding
        # df = one_hot_encode_columns(df, categorical_columns)

        # Encode the target variable
        df["risk"].replace({"bad": 1, "good": 0}, inplace=True)

        return df

    def preprocess_data_inference(self, df):
        """Preprocessing inference data - quite similar to training
        but with modifications such as
        using the trained One-hot encoder,
        filling missing values with the most frequent category as obtained from training file"""
        df = df.toPandas()

        # Similar as the training file inference
        try:
            df = df.drop("_c0", axis=1)
        except KeyError:
            pass

        # Define the columns to be one-hot encoded
        columns_to_encode = [
            "purpose",
            "sex",
            "housing",
            "savings_account",
            "checking_account",
            "age_group",
            "credit_amount_range",
        ]

        # Load the joblib file using the full path
        ohe = joblib.load(os.path.join(cwd, "utils/ohe.joblib"))

        array_hot_encoded = ohe.transform(df[columns_to_encode])
        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df.index)
        data_other_cols = df.drop(columns=columns_to_encode)

        df = pd.concat([data_hot_encoded, data_other_cols], axis=1)
        return df


