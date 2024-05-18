import joblib

# from utils.training_utils import
from .training_utils import TrainingUtils

from fastapi import File

class Infer_Credit:

    def __init__(self):
        self.trainer = TrainingUtils()

    # Example usage within an Airflow task
    def process_credit_data_inference(self, input_file: File): #type:ignore
        spark = self.trainer.create_spark_session("CreditDataProcessing")
        df_credit = self.trainer.load_dataset(spark, input_file) # read the input file
        
        print(f'read the input file nicely, it is {df_credit}')


        df_credit = self.trainer.rename_columns(df_credit)
        df_credit = self.trainer.rename_specific_columns(df_credit)
        # most_freq_savings = trainer.get_most_frequent_category(df_credit, "savings_account")
        # most_freq_checking = trainer.get_most_frequent_category(
        #     df_credit, "checking_account"
        # )

        print(f'before fill missing is {df_credit}')

        df_credit = self.trainer.fill_missing_values(df_credit, "savings_account", "little")
        df_credit = self.trainer.fill_missing_values(df_credit, "checking_account", "moderate")
        
        print(f'after fill missing is {df_credit}')

        df_credit = self.trainer.create_new_features(df_credit)



        # Additional processing can be added here

        df_credit = self.trainer.preprocess_data_inference(df_credit)
        return df_credit


    def make_inferences(self, input_file: File):  # type:ignore
        df = self.process_credit_data_inference(input_file)
        df.columns = df.columns.astype(str)

        # Use the loaded model to make predictions
        loaded_model = joblib.load("random_forest.pkl")


        result = loaded_model.predict(df)

        return result
