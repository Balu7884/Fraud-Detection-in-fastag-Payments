import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report

# Load the data
def load_data(file_path='data/cleaned_data.csv'):
    df = pd.read_csv(file_path)
     # Feature Engineering
    df['Amount_Mismatch'] = df['Transaction_Amount'] - df['Amount_paid']
    df['Vehicle_Profile'] = df['Vehicle_Type'] + '_' + df['Vehicle_Dimensions']
    df['High_Speed'] = df['Vehicle_Speed'] > 80  # Boolean feature

    # One-hot encode relevant categorical columns
    df_encoded = pd.get_dummies(df[['Lane_Type', 'Vehicle_Profile', 'State_code']], drop_first=True)

    # Combine numeric and encoded features
    X = pd.concat([
        df[['Amount_Mismatch', 'Vehicle_Speed', 'High_Speed']],
        df_encoded
    ], axis=1)

    # Target variable
    y = df['Fraud_indicator']
    
    return X, y

# Train-test split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Train Random Forest Model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest Model Trained.")
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Random Forest Model: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# Save Model and Label Encoders
def save_model(model, model_path='random_forest_model.joblib'):#encoder_path='label_encoders.joblib'
    joblib.dump(model, model_path)
    # joblib.dump(encoder_path)
    print(f"Model saved to {model_path}")

# Main Execution
def main():
    X, y = load_data()

    # Save feature names used in training
    model_features = X.columns.tolist()
    joblib.dump(model_features, 'model_features.joblib')
    print("✅ Model features saved to 'model_features.joblib'",model_features)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()
