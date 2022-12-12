# Script to train machine learning model.
from helper_functions import load_data, process_data, train_model, inference
from helper_functions import measure_metrics, split_data, measure_slice_metrics

# Add code to load in the data.
df = load_data(filepath='data/census.csv')

# Split data
train, test = split_data(df)

# Train the model
X_train, y_train = process_data(train, label="salary")
X_train, y_train, encoder, scaler, model = train_model(X_train, y_train)

# Run inference
X_test, y_test = process_data(test, label="salary")
y_pred = inference(model, X_test, encoder, scaler)

# Measure slice accuracy for distinct race column values
race_distinct_vals = X_test.race.unique()
with open('slice_output.txt', 'w') as f:
    for val in race_distinct_vals:
        slice_accuracy = measure_slice_metrics(
            X_test, y_test, 'race', val, model, encoder, scaler)
        slice_accuracy = round(slice_accuracy, 3)
        line = f"{val} slice accuracy: {slice_accuracy}"
        f.write(line)
        f.write('\n')
