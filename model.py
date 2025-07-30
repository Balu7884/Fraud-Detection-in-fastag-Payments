import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.pipeline import make_pipeline

# Load Cleaned Data
def load_data(file_path='data/cleaned_data.csv'):
    df = pd.read_csv(file_path)
    # Amount Mismatch
    #df['Amount_Mismatch'] = df['Transaction_Amount'] - df['Amount_paid']

    # Vehicle Profile
    #df['Vehicle_Profile'] = df['Vehicle_Type'] + '_' + df['Vehicle_Dimensions']

    # High-Speed Flag (optional threshold e.g., 80 km/h)
    df['High_Speed'] = df['Vehicle_Speed'] > 80
    return df
def load_and_preprocess_data(file_path='data/cleaned_data.csv'):
    """
    Loads, encodes, and splits the data by calling the helper functions.
    This function is intended to be imported by the Streamlit app.
    """
    # 1. Load the initial data
    df = load_data(file_path)
    
    # 2. Apply label encoding
    df, label_encoder = label_encode_data(df)
    
    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Return all the components needed by the app
    return df, label_encoder, X_train, X_test, y_train, y_test


# Label Encoding
def label_encode_data(df):
    label_encoder = {}
    object_columns = ['Vehicle_Type', 'Lane_Type','State_code','TollBoothID']
    for column in object_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoder[column] = le
    return df, label_encoder

# Split Data into Train and Test
def split_data(df):
    X = df.drop(columns=["Fraud_indicator"])
    y = df["Fraud_indicator"]
    print(X)
    print(y)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate Model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1, conf_matrix

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
# XGBoost Classifier
def xgboost_classifier(X_train, X_test, y_train, y_test):
   
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(objective='binary:logistic',
                              eval_metric='logloss',
                              use_label_encoder=False,
                              scale_pos_weight=scale_pos_weight,
                              random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

# Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Logistic Regression Results:")
    return evaluate_model(y_test, y_pred)

# Decision Tree
def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Decision Tree Results:")
    return evaluate_model(y_test, y_pred)

# SVM Classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # print("Support Vector Machine Classifier Results:")
    return evaluate_model(y_test, y_pred)

# Random Forest Classifier
def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200],         
        'max_depth': [10, 20, None],        
        'min_samples_leaf': [1, 2, 4]       
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='f1')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    return evaluate_model(y_test, y_pred)

# KNN Classifier
def knn_classifier(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("KNN Classifier Results:")
    return evaluate_model(y_test, y_pred)

def voting_classifier(X_train, X_test, y_train, y_test):
    """
    Creates and evaluates a Voting Classifier that combines multiple models.
    
    This ensemble model uses 'soft' voting, which averages the predicted
    probabilities from the base models for a more nuanced final prediction.
    """
    # 1. Define the base models that will be part of the ensemble.
    # We use a pipeline for SVM to ensure the data is scaled correctly for that model.
    clf1 = LogisticRegression(random_state=42,class_weight='balanced',max_iter=1000)
    clf2 = RandomForestClassifier( random_state=42, class_weight='balanced', n_estimators=100)
    
    # SVC needs probability=True to be used in a soft voting ensemble.
    clf3 = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))

    # 2. Create the Voting Classifier
    # We name each model ('lr', 'rf', 'svc') and combine them.
    ensemble_model = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='soft'  # 'soft' voting uses predicted probabilities and often performs better.
    )

    # 3. Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # 4. Make predictions and evaluate
    y_pred = ensemble_model.predict(X_test)
    
    # The evaluate_model function you already have will calculate accuracy, etc.
    return evaluate_model(y_test, y_pred)

# Model Comparison Plot
def plot_model_comparison(results, model_names):
    """
    Plots a bar chart comparing the accuracy scores of different models.
    
    Args:
        results (list): A list of tuples, where each tuple contains model metrics.
                        The accuracy score should be the first element (index 0).
        model_names (list): A list of strings with the names of the models.
    """
    # Extract accuracy scores from the results list
    accuracy_scores = [res[0] for res in results]
    
    # Dynamically generate a color for each model bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

    # Create Figure and Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use the 'model_names' list passed into the function
    bars = ax.bar(model_names, accuracy_scores, color=colors)

    ax.set_xlabel('Machine Learning Models', fontsize=12)
    ax.set_ylabel('Accuracy Scores', fontsize=12)
    ax.set_title('Comparison of Model Accuracies', fontsize=16)
    
    # Add the accuracy value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    # Rotate labels to prevent them from overlapping
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0) # Set y-axis limit from 0 to 1 for accuracy
    plt.tight_layout() # Adjust layout to make sure everything fits

    return fig  # Return the figure instead of plt.show()