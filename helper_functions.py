import pandas as pd
import numpy as np
import category_encoders as ce
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def split_data(df, test_size=0.2, stratify=False):
    if stratify != False:
        train, test = train_test_split(
            df, test_size=test_size, stratify=stratify)
    else:
        train, test = train_test_split(df, test_size=test_size)

    return train, test


def process_data(df, label):
    # Remove unnecessary spaces from categorical data
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: x.replace(" ", ""))

    # Convert salary into a numerical (binary) column
    df[label] = df[label].apply(lambda x: 1 if x == '>50K' else 0)

    # Drop columns that dont have predictive value
    df = df.drop(['fnlgt', 'education', 'relationship'], axis=1)

    # Split X and y
    X_train = df.drop(label, axis=1)
    y_train = df[label]

    return X_train, y_train


def train_model(X_train, y_train):
    categorical_features = X_train.select_dtypes(include='object').columns

    # use one-hot encoding to encode categorical features
    onehot_encoder = OneHotEncoder(cols=categorical_features)
    X_train = onehot_encoder.fit_transform(X_train)

    # fit and transform scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    return X_train, y_train, onehot_encoder, scaler, model


def inference(model, X_test, encoder, scaler):
    X_test = encoder.transform(X_test)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    return y_pred


def measure_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def measure_slice_metrics(X_test, y_test, column, value, model, encoder, scaler):
    X_slice = X_test[X_test[column] == value]
    y_slice = y_test[X_slice.index]
    y_slice_pred = inference(model, X_slice, encoder, scaler)
    accuracy = measure_metrics(y_slice, y_slice_pred)

    return accuracy


def output_slice_metrics(X_test, y_test, col, distinct_values, model, encoder, scaler):
    with open('slice_output.txt', 'w') as f:
        for val in distinct_values:
            slice_accuracy = measure_slice_metrics(
                X_test, y_test, col, val, model, encoder, scaler)
            slice_accuracy = round(slice_accuracy, 3)
            line = f"{val} slice accuracy: {slice_accuracy}"
            f.write(line)
            f.write('\n')
