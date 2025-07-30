import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import stats
import live_stats
import joblib
from model import (
    load_and_preprocess_data, 
    logistic_regression, decision_tree, random_forest, svm_classifier, knn_classifier,
    xgboost_classifier, voting_classifier,
    plot_model_comparison
)

model = joblib.load('random_forest_model.joblib')
model_features = joblib.load('model_features.joblib')


# Load the data
df_banks_list = pd.read_csv("data/netc_banks_data.csv") #NETC banks data
cleaned_data = pd.read_csv("data/cleaned_data.csv") #Cleaned data
df = pd.read_csv("data/NETC_Monthly_Transactions.csv")#NETC monthly transaction data
df['Date'] = pd.to_datetime(df['Date']) #Applying Modifications to NETC monthly transaction data
df_yearly = pd.read_csv("data/netc_yearly_data.csv") #NETC yearly data
dataframe = pd.read_csv('data/FastagFraudDetection.csv')
dataframe = stats.preprocess_data(dataframe)

# Load the trained model and label encoders
model = joblib.load('random_forest_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# ---- Streamlit UI ----
# ---- State Management for Navigation ----
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---- Sidebar with Buttons ----
st.sidebar.title("Fastag Fraud Detection System")
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Models"):
    st.session_state.page = "Models"
if st.sidebar.button("Real Time Statistics"):
    st.session_state.page = "Real Time Statistics"
if st.sidebar.button("Statistics"):
    st.session_state.page = "Statistics"


# ---- Page Handling ----
if st.session_state.page == "Home":
    st.title("Welcome to Fastag Fraud Detection")
    st.header("Predict Fraud Transactions")
    option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
    if option == "Manual Input":
        st.subheader("Enter Transaction Details:")

        vehicle_types = cleaned_data['Vehicle_Type'].unique()
        Toll_booth_ids = cleaned_data['TollBoothID'].unique()
        lane_types = cleaned_data['Lane_Type'].unique()
        #vehicle_dimensions_select = cleaned_data['Vehicle_Dimensions'].unique()
        dayOfWeek_select = cleaned_data['DayOfWeek'].unique()
        state_code_select = cleaned_data['State_code'].unique()

        # User Inputs
        vehicle_type = st.selectbox("Select Vehicle Type", vehicle_types)
        tollbooth_id = st.selectbox("Toll Booth ID", Toll_booth_ids)
        lane_type = st.selectbox("Lane Type", lane_types)
        #vehicle_dimensions = st.selectbox("Vehicle Dimensions", vehicle_dimensions_select)
        transaction_amount = st.number_input("Transaction Amount", min_value=0)
        amount_paid = st.number_input("Amount Paid", min_value=0)
        vehicle_speed = st.number_input("Vehicle Speed (km/h)", min_value=0)
        hour = st.slider("Hour of Transaction", 0, 23, 12)
        day_of_week = st.selectbox("Day of the Week", dayOfWeek_select)
        state_code = st.selectbox("State Code", state_code_select)

        if st.button("Predict Fraud"):
            # Feature Engineering
            amount_mismatch = transaction_amount - amount_paid
            high_speed = vehicle_speed > 80
            #vehicle_profile = f"{vehicle_type}_{vehicle_dimensions}"

            # Create a one-row DataFrame
            input_data = pd.DataFrame([{
                'Amount_Mismatch': amount_mismatch,
                'Vehicle_Speed': vehicle_speed,
                'High_Speed': high_speed,
                'Lane_Type': lane_type,
                #'Vehicle_Profile': vehicle_profile,
                'State_code': state_code
            }])

            # Get dummy columns from training data
            dummy_cols = [col for col in model_features if col not in ['Amount_Mismatch', 'Vehicle_Speed', 'High_Speed']]

            # Apply one-hot encoding
            input_encoded = pd.get_dummies(input_data, columns=['Lane_Type', 'State_code'])

            # Add missing columns (columns that were in training but not in this input)
            for col in dummy_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0  # Add with 0 default

            # Ensure order of columns matches training
            input_encoded = input_encoded[model_features]

            # Predict
            prediction = model.predict(input_encoded)[0]
            prediction_label = "Fraud Detected" if prediction == 1 else "Not Fraud"

            st.write("### Prediction Result:")
            st.write(f"**{prediction_label}**")

    elif option == "Upload CSV":
        st.subheader("📂 Upload CSV File")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", df.head())

            required_columns = ["Vehicle_Type", "TollBoothID", "Lane_Type",
                                "Transaction_Amount", "Amount_paid", "Vehicle_Speed", "Hour", "DayOfWeek", "State_code"]

            if all(col in df.columns for col in required_columns):
                # Load model and model features
                model = joblib.load('random_forest_model.joblib')
                model_features = joblib.load('model_features.joblib')

                # Feature Engineering
                df['Amount_Mismatch'] = df['Transaction_Amount'] - df['Amount_paid']
                #df['Vehicle_Profile'] = df['Vehicle_Type'] + '_' + df['Vehicle_Dimensions']
                df['High_Speed'] = df['Vehicle_Speed'] > 80

                # One-hot encode relevant columns
                df_encoded = pd.get_dummies(df[['Lane_Type', 'State_code']], drop_first=True)

                # Combine all features
                X = pd.concat([
                    df[['Amount_Mismatch', 'Vehicle_Speed', 'High_Speed']],
                    df_encoded
                ], axis=1)

                # Ensure all columns match the trained model's features
                for col in model_features:
                    if col not in X.columns:
                        X[col] = 0  # Add missing column with default 0

                # Ensure the correct column order
                X = X[model_features]

                # Make Predictions
                predictions = model.predict(X)
                df['Fraud_Prediction'] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]

                # Output
                st.write("Predictions:")
                st.write(df)

                st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")
            else:
                st.error("Uploaded file is missing required columns!")


elif st.session_state.page == "Models":
    st.title("Machine Learning Model Evaluation")

    # Load and Preprocess Data
    df, label_encoder, X_train, X_test, y_train, y_test = load_and_preprocess_data()

    results = []

    st.success("Model Performance")
for name, model_func in zip(
    ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "KNN", "XGBoost","Voting Classifier"],
    [logistic_regression, decision_tree, random_forest, svm_classifier, knn_classifier, xgboost_classifier,voting_classifier]
):
    st.write(f"---") # Add a separator for clarity
    st.write(f"### {name} Results")

    # This part remains the same
    accuracy, precision, recall, f1, conf_matrix = model_func(X_train, X_test, y_train, y_test)
    results.append((accuracy, precision, recall, f1, conf_matrix))

    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy:.2f}")
    with col2:
        st.metric(label="Precision", value=f"{precision:.2f}")
    with col3:
        st.metric(label="Recall", value=f"{recall:.2f}")
    with col4:
        st.metric(label="F1 Score", value=f"{f1:.2f}")
    
    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Model Comparison Plot
st.subheader("Comparison of Model Accuracies")
model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "KNN", "XGBoost", "Voting Classifier"]
fig = plot_model_comparison(results, model_names) # Pass both results and the names
st.pyplot(fig)

if st.session_state.page == "Real Time Statistics":
    st.title("Real Time Statistics")
    st.write(
        "Welcome to the Real-Time Fraud Detection Dashboard. This dashboard presents insightful analysis using data sourced from the National Payments Corporation of India (NPCI) and official reports. "
    )
    st.header("National Electronic Toll Collection Banks Data")
    st.dataframe(df_banks_list)
    st.info("As per the directive of Reserve Bank of India (RBI) dated on 24th Sep 2019 operations of PMC Bank has been stopped.")
    st.warning("As per the directive of Reserve Bank of India RBI dated 31st January 2024, customers are permitted to use their FASTag without any restrictions, up to their available balance.")
    st.info("As per the directive of Reserve Bank of India (RBI) dated on 31st January 2024, operations of PAYTM Payments Bank has been stopped.")
    st.header("NETC Monthly Transactions data")
    st.dataframe(df[['Volume (Mn)', 'Value (Cr)']].describe())
    fig,ax = live_stats.plot_trend_analysis(df)
    st.pyplot(fig)
    fig,axes = live_stats.plot_distribution_analysis(df)
    st.pyplot(fig)
    st.title("NETC Yearly Transactions data")
    st.header("Summary")
    st.dataframe(df_yearly)
    st.dataframe(df_yearly.describe())
    fig = live_stats.plot_trend_analysis_yearly()
    st.pyplot(fig)
    fig = live_stats.plot_correlation_heatmap(df_yearly)
    st.pyplot(fig)

elif st.session_state.page == "Statistics":
    fraud_count,top_state, top_fraud_count,highest_month_name,highest_month_value,highest_day_name,highest_day_value = stats.stats_csv(dataframe)
    st.title("Welcome to the Fraud Detection Analysis")
    st.header("Top Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("Total Frauds")
        st.error(fraud_count)
        
    with col2:
        st.markdown("State with Highest Frauds")
        st.warning(f"**{top_state}** : **{top_fraud_count}** ")
    with col3:
        st.markdown("Month-wise Highest frauds")
        st.success(f"**{highest_month_name}** : **{highest_month_value}**")
    with col4:
        st.markdown("Day-wise Highest Frauds")
        st.info(f"**{highest_day_name}** : **{highest_day_value}**")
    # Pie chart of fraud percentage
    st.header("Fraud Percentage")    
    fig, ax = stats.fraud_chart(dataframe)
    st.pyplot(fig)

    st.title("Data Analysis - KDE and Correlation Heatmap")
    (fig1, ax1), (fig2, ax2) = stats.generate_plots(dataframe)
    col1, col2 = st.columns(2)
    with col1:
        # KDE Plot for Transaction Amount and Amount Paid
        st.write("### Kernel Density Estimation Plot")
        st.pyplot(fig1)
    with col2:
        # Correlation matrix and heatmap for numerical variables
        st.write("Correlation matrix and heatmap")
        st.pyplot(fig2)
    
    st.title("Transaction Analysis - Histograms")
    fig1, ax1, fig2, ax2 = stats.plot_histograms(dataframe)
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Histogram of Transaction Amount")
        st.pyplot(fig1)
    with col2:
        st.write("### Histogram of Amount Paid")
        st.pyplot(fig2)

    st.title("Regression and Scatter Plots")
    col1,col2 = st.columns(2)
    with col1:
        # Regression Plot
        st.write("### Regression Plot: Transaction Amount vs Amount Paid")
        fig1, ax1 = stats.plot_regression(dataframe)
        st.pyplot(fig1)
    with col2:
        # Scatter Plot
        st.write("### Scatter Plot: Fraud vs Not Fraud")
        fig2, ax2 = stats.plot_scatter(dataframe)
        st.pyplot(fig2)

    st.title("Fraud Detection Analysis")
    # Display the plots
    st.subheader("Fraud Indicator Analysis")
    fig = stats.plot_fraud_indicators(dataframe)
    st.pyplot(fig)

    st.title("Fraud Detection Dashboard")
    col1,col2 = st.columns(2)
    with col1:
        # Visualize Day-wise Fraud Activity
        st.subheader("Day-wise Fraud Activity")
        fig_daywise = stats.plot_daywise_fraud(dataframe)
        st.pyplot(fig_daywise)
    with col2:
        # Visualize Monthly Transactions
        st.subheader("Month-Wise Fraud Activity")
        fig_monthly = stats.plot_monthly_transactions(dataframe)
        st.pyplot(fig_monthly)

    col1,col2 = st.columns(2)
    with col1:
        # Visualize Day-wise Fraud Activity
        st.subheader("State-Wise Fraud Activity")
        fig_statewise = stats.plot_statewise_fraud(dataframe)
        st.pyplot(fig_statewise)
    with col2:
         # Visualize Monthly Transactions
        pass