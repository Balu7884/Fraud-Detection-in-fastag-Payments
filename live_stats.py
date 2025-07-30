import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_yearly = pd.read_csv("data/netc_yearly_data.csv")

def plot_trend_analysis(data):
    st.header("Trend Analysis: Volume and Value Over Time")
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Volume (Mn)'], label='Volume (Mn)', color='blue')
    ax.plot(data['Date'], data['Value (Cr)'], label='Value (Cr)', color='green')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.set_title('Trend Analysis: Volume and Value over Time')
    ax.legend()
    ax.grid(True)
    
    return fig,ax

def plot_distribution_analysis(data):
    st.title("Distribution Analysis: Volume and Value")

    # Plot using Matplotlib and Seaborn
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(data['Volume (Mn)'], kde=True, color='blue', ax=axes[0])
    axes[0].set_title('Distribution of Volume (Mn)')
    
    sns.histplot(data['Value (Cr)'], kde=True, color='green', ax=axes[1])
    axes[1].set_title('Distribution of Value (Cr)')

    plt.tight_layout()
    return fig,axes

def plot_trend_analysis_yearly():
    st.subheader("Trend Analysis: Volume and Amount Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot( df_yearly['Month'], label='Volume (In Mn)', color='blue', marker='.')
    ax.plot(df_yearly['Month'], df_yearly['Amount (In Cr)'], label='Amount (In Cr)', color='green',marker='.')
    ax.set_xlabel('Month')
    ax.set_ylabel('Values')
    ax.legend()
    plt.xticks(rotation=90)
    ax.grid(True)
    return fig

def plot_correlation_heatmap(df):
    st.subheader("🔎 Correlation Analysis")
    correlation = df[['No. of Banks Live on NETC', 'Tag Issuance (In Nos.)', 'Volume (In Mn)', 'Amount (In Cr)']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    return fig
