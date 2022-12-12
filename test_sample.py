import pytest
import pandas as pd
from helper_functions import load_data, process_data, train_model, inference, measure_metrics, split_data


@pytest.fixture(scope="session")
def df():
    data = pd.read_csv("./data/census.csv")
    return data


def test_split_data(df):
    train, test = split_data(df)
    assert train.shape[0] > 0


def test_process_data(df):
    train, test = split_data(df)
    X_train, y_train = process_data(train, label="salary")

    # Check columns in X
    assert 'salary' not in X_train.columns
    assert 'age' in X_train.columns

    # Make sure not all classes in the label are the same
    assert len(y_train) > 0

    # Make sure not all classes in the label are the same
    assert y_train.mean() > 0


def test_prediction(df):
    train, test = split_data(df)
    X_train, y_train = process_data(train, label="salary")
    X_train, y_train, encoder, scaler, model = train_model(X_train, y_train)
    X_test, y_test = process_data(test, label="salary")
    y_pred = inference(model, X_test, encoder, scaler)

    # Check that predictions are created successfully
    assert len(y_pred) > 0
