import os
import pickle
import socket
import sys
import time

import pandas as pd


data = pd.read_csv("./trial_data/sample.csv", index_col=0)

# drop target col
data = data.drop(["risk"], axis=1)
print(data.shape)

model = pickle.load(open("./app/model.pkl", "rb"))

predictions = model.predict(data)

print(predictions)