import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("car_evaluation.csv")

# Split data into features (X) and target variable (y)
X = data.drop(columns=['decision'])
y = data['decision']

# Preprocessing: Convert categorical variables into numerical format
X_encoded = pd.get_dummies(X)  # One-hot encoding for categorical variables

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, "trained_car_evaluation_model_rf.pkl")

# Save the mapping used for one-hot encoding to a file
one_hot_encoding_mapping = X_encoded.columns.tolist()
joblib.dump(one_hot_encoding_mapping, "one_hot_encoding_mapping.pkl")