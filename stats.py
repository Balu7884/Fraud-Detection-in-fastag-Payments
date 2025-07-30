import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

palette = {'Fraud': 'red', 'Not Fraud': 'blue'}

def stats_csv(dataframe):
    fraud_count = dataframe["Fraud_indicator"].value_counts().get("Fraud", 0)

    fraud_data = dataframe[dataframe['Fraud_indicator'] == 'Fraud'] # Filter for fraud cases
    state_fraud_count = fraud_data.groupby('State_code')['Fraud_indicator'].count() # Group by State_code and count frauds
    # Find the top state
    top_state = state_fraud_count.idxmax()
    top_fraud_count = state_fraud_count.max()

    # Month-wise fraud count
    month_fraud_count = fraud_data.groupby('Month')['Fraud_indicator'].count()
    highest_month = month_fraud_count.idxmax()
    highest_month_value = month_fraud_count.max()

    day_fraud_count = fraud_data.groupby('DayOfWeek')['Fraud_indicator'].count()
    highest_day = day_fraud_count.idxmax()
    highest_day_value = day_fraud_count.max()

    # Mapping DayOfWeek to actual day names
    days_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    month_mapping = {
        1:"January",  2:"February",  3:"March",  4:"April",  5:"May",  6:"June",  8:"July",  9:"August", 10:"September",  7:"October", 11:"November", 12:"December",
    }

    highest_day_name = days_mapping[highest_day]
    highest_month_name = month_mapping[highest_month]

    return fraud_count,top_state, top_fraud_count,highest_month_name,highest_month_value,highest_day_name,highest_day_value

def fraud_chart(dataframe):
    fraud_values = dataframe["Fraud_indicator"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5)) 
    ax.pie(fraud_values, labels=fraud_values.index, autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.axis('equal')  # Ensures pie chart is a circle
    ax.set_title("Fraud vs Not Fraud Distribution")
    return fig, ax

def generate_plots(dataframe):
    # --- Plotting KDE Plot ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=dataframe['Transaction_Amount'], fill=True, label='Transaction Amount', ax=ax1)
    sns.kdeplot(data=dataframe['Amount_paid'], fill=True, label='Amount Paid', ax=ax1)
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Density')
    ax1.set_title('Kernel Density Estimation of Transaction Amount and Amount Paid')
    ax1.legend()

    # --- Plotting Correlation Heatmap ---
    correlation_matrix = dataframe[['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Correlation Heatmap')

    return (fig1, ax1), (fig2, ax2)

def plot_histograms(dataframe):
    # Plotting histogram for Transaction Amount
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(dataframe['Transaction_Amount'], bins=30, edgecolor='black')
    ax1.set_xlabel('Transaction Amount')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Transaction Amount')

    # Plotting histogram for Amount Paid
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(dataframe['Amount_paid'], bins=30, edgecolor='black')
    ax2.set_xlabel('Amount Paid')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Amount Paid')

    return fig1, ax1, fig2, ax2

def plot_regression(dataframe):
    # Plotting the regression plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.regplot(x='Transaction_Amount', y='Amount_paid', data=dataframe, ax=ax1,line_kws={'color': 'green'})
    ax1.set_title('Regression Plot: Transaction Amount vs Amount Paid')
    ax1.set_xlabel('Transaction Amount')
    ax1.set_ylabel('Amount Paid')

    return fig1, ax1

def plot_scatter(dataframe):
    # Plotting the scatter plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=dataframe, 
                     x="Transaction_Amount", 
                     y="Amount_paid", 
                     hue="Fraud_indicator",  
                     palette=palette,
                     ax=ax2)
    ax2.set_title('Scatter Plot: Transaction Amount vs Amount Paid (Fraud Detection)')
    ax2.set_xlabel('Transaction Amount')
    ax2.set_ylabel('Amount Paid')
    ax2.grid(True)

    return fig2, ax2

def plot_fraud_indicators(dataframe):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Vehicle Types vs Fraud Indicator
    sns.countplot(x="Vehicle_Type", data=dataframe, hue="Fraud_indicator", ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Vehicle Type vs Fraud Indicator")

    # Lane Types vs Fraud Indicator
    sns.countplot(x="Lane_Type", data=dataframe, hue="Fraud_indicator", ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("Lane Type vs Fraud Indicator")

    # Different Toll Booths vs Fraud Indicator
    sns.countplot(x="TollBoothID", data=dataframe, hue="Fraud_indicator", ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Toll Booths vs Fraud Indicator")

    # Vehicle Dimensions vs Fraud Indicator
    sns.countplot(x="Vehicle_Dimensions", data=dataframe, hue="Fraud_indicator", ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Vehicle Dimensions vs Fraud Indicator")

    plt.tight_layout()
    return fig

def preprocess_data(dataframe):
    dataframe["State_code"] = dataframe["Vehicle_Plate_Number"].str[:2]
    # Dictionary mapping state codes to full names
    state_mapping = {
        'KA': 'Karnataka',
        'MH': 'Maharashtra',
        'AP': 'Andhra Pradesh',
        'GA': 'Goa',
        'KL': 'Kerala',
        'GJ': 'Gujarat',
        'TN': 'Tamil Nadu',
        'DL': 'Delhi',
        'TS': 'Telangana',
        'UP': 'Uttar Pradesh',
        'RJ': 'Rajasthan',
        'WB': 'West Bengal',
        'MP': 'Madhya Pradesh',
        'HR': 'Haryana',
        'BR': 'Bihar'
    }

    # Map state codes to state names
    dataframe['State_code'] = dataframe['State_code'].map(state_mapping)

    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])
    dataframe["Hour"] = dataframe["Timestamp"].dt.hour
    dataframe["DayOfWeek"] = dataframe["Timestamp"].dt.dayofweek
    dataframe["Month"] = dataframe["Timestamp"].dt.month
    dataframe[['Latitude', 'Longitude']] = dataframe['Geographical_Location'].str.split(',', expand=True).astype(float)
    
    return dataframe

def plot_daywise_fraud(dataframe):
    # Visualize Fraud Activity by Day
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=dataframe, x="DayOfWeek", hue="Fraud_indicator", palette=palette, ax=ax)
    ax.set_title("Day of Fraud Activity")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return fig

def plot_monthly_transactions(dataframe):
    dataframe.set_index('Timestamp', inplace=True) # Set Timestamp as index
    monthly_fraud_data = dataframe.resample('M')['Fraud_indicator'].value_counts().unstack().fillna(0) # Calculate fraud and not fraud transactions
    fig, ax = plt.subplots(figsize=(12, 7))
    monthly_fraud_data.plot(kind='bar', stacked=False, color=['red', 'blue'], ax=ax)
    month_names = monthly_fraud_data.index.strftime('%b %Y')  # 'Jan 2025', 'Feb 2025', etc.
    ax.set_xticks(range(len(month_names)))
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    ax.set_title('Monthly Fraud and Not Fraud Transactions')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Transactions')
    ax.legend(loc='upper right', title='Transaction Type')
    ax.grid(True)
    
    return fig

def plot_statewise_fraud(dataframe):
    sns.set_theme(style="whitegrid")
    palette = {'Fraud': 'red', 'Not Fraud': 'blue'}
    
    # Plotting using Seaborn
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=dataframe, x="State_code", hue="Fraud_indicator", palette=palette, ax=ax)

    # Set plot details
    ax.set_title("Fraud Transaction Count based on State Codes")
    ax.set_xlabel("State Code")
    ax.set_ylabel("Count of Transactions")
    ax.legend(title="Fraud")
    
    return fig
