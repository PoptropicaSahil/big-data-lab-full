import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from utils.training_utils import
from .training_utils import TrainingUtils


class TRAIN_MODEL:
    def __init__(self):
        self.trainer = TrainingUtils()



    # Example usage within an Airflow task
    def process_credit_data_training(self):
        spark = self.trainer.create_spark_session("CreditDataProcessing")
        df_credit = self.trainer.load_dataset(spark, "german_credit_data.csv")
        df_credit = self.trainer.rename_columns(df_credit)
        df_credit = self.trainer.rename_specific_columns(df_credit)
        most_freq_savings = self.trainer.get_most_frequent_category(df_credit, "savings_account")
        most_freq_checking = self.trainer.get_most_frequent_category(
            df_credit, "checking_account"
        )
        df_credit = self.trainer.fill_missing_values(
            df_credit, "savings_account", most_freq_savings
        )
        df_credit = self.trainer.fill_missing_values(
            df_credit, "checking_account", most_freq_checking
        )
        df_credit = self.trainer.create_new_features(df_credit)
        # Additional processing can be added here

        df_credit = self.trainer.preprocess_data_training(df_credit)
        return df_credit
    

    def train_and_dump_model(self):

        df = self.process_credit_data_training()
        df.columns = df.columns.astype(str)

        X = df.drop("risk", axis=1)
        y = df["risk"]
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )


        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, conf, betaf = self.trainer.eval_metrics(y_test, y_pred)
        print("Accuracy", {acc})

        # Save the model as a pickle in a file
        joblib.dump(model, "random_forest.pkl")
