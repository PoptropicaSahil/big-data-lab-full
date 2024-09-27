import os
import tempfile

import joblib
from fastapi import File

from .training_utils import TrainingUtils

cwd = os.getcwd()


class Infer_Credit:
    def __init__(self):
        self.trainer = TrainingUtils()

    # Process the data for inference
    def process_credit_data_inference(self, input_file: File):  # type:ignore
        """Process the data for inference - this calls methods from the trainer class"""

        spark = self.trainer.create_spark_session("CreditDataProcessing")

        
        ### Boundary Check for reading the file
        try:
            # Read the data by storing in a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                # Write the contents of the File object to the temporary file
                temp.write(input_file.file.read())
                temp.flush()
                temp_path = temp.name

            # Read the temporary file using Spark
            df_credit = spark.read.csv(temp_path, header=True, inferSchema=True)
        except Exception as e:
            print(f'ERROR READING FILE. EXCEPTION IS {e}')

        
        data_validation_check_flag = self.trainer.data_val_checks(df_credit)

        if data_validation_check_flag == 0:
            print('DATA VALIDATION SUCCESSFUL')
        else:
            raise Exception('''ENTERED DATTA IS NOT IN EXPECTED FORMAT. PLEASE CHECK SAMPLE INFERENCE FILE FOR REFERENCE
                            All given columns should be present and the columns except Saving accounts and Checking account
                            should not have any missing values''')

        df_credit = self.trainer.rename_columns(df_credit)
        df_credit = self.trainer.rename_specific_columns(df_credit)

        df_credit = self.trainer.fill_missing_values(
            df_credit, "savings_account", "little"
        )
        df_credit = self.trainer.fill_missing_values(
            df_credit, "checking_account", "moderate"
        )

        df_credit = self.trainer.create_new_features(df_credit)
        # Additional processing can be added here
        df_credit = self.trainer.preprocess_data_inference(df_credit)
        return df_credit

    def make_inferences(self, input_file: File):  # type:ignore
        """The method that is called for making inferences on the user data"""
        df = self.process_credit_data_inference(input_file)
        df.columns = df.columns.astype(str)

        # Use the loaded model to make predictions
        loaded_model = joblib.load(os.path.join(cwd, "utils/random_forest.pkl"))

        result = loaded_model.predict(df)

        return result
