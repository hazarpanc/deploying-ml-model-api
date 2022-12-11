# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from helper_functions import load_data, process_data

# Add code to load in the data.
df = load_data(filepath = 'data/census.csv')

train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
