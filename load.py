import pandas as pd
import joblib

# Load the trained model from the saved file
model = joblib.load("trained_car_evaluation_model_rf.pkl")

# Accept user input for car attributes
buying = input("Enter buying price (vhigh, high, med, low): ")
maintenance = input("Enter maintenance cost (vhigh, high, med, low): ")
doors = input("Enter number of doors (2, 3, 4, 5more): ")
persons = input("Enter number of persons (2, 4, more): ")
lug_boot = input("Enter size of luggage boot (small, med, big): ")
safety = input("Enter safety level (low, med, high): ")

# Create a DataFrame with user input
input_data = pd.DataFrame({
    'buying': [buying],
    'maintenance': [maintenance],
    'doors': [doors],
    'persons': [persons],
    'lug_boot': [lug_boot],
    'safety': [safety]
})

# Load the original mapping used for one-hot encoding during training
mapping = joblib.load("one_hot_encoding_mapping.pkl")

# Apply one-hot encoding to the input data using the original mapping
input_data_encoded = pd.get_dummies(input_data).reindex(columns=mapping, fill_value=0)

# Make predictions on the input data
predictions = model.predict(input_data_encoded)

# Display the predicted decision
print("Predicted Decision:", predictions[0])            