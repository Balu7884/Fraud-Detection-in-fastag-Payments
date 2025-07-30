
# elif st.session_state.page == "Text":

#     st.title("🚀 Streamlit Text Methods `st.title`")

#     st.header("Basic Text Display `st.header`")
#     st.text("This is plain text using `st.text()`")
#     st.write("This text is using **st.write()** which can handle markdown, dataframes, and more.`st.write`")

#     st.header("Markdown & Captions")
#     st.markdown("### This is a Markdown Header using `st.markdown()`")
#     st.caption("This is a small caption using `st.caption()`")

#     st.header("Code & Math Display")
#     st.code("print('Hello, Streamlit!') `st.code`", language='python')
#     st.latex(r"E = mc^2 `st.latex`")

#     st.header("Message Alerts")
#     st.success("✅ Success message example `st.success`")
#     st.error("❌ Error message example")
#     st.warning("⚠️ Warning message example")
#     st.info("ℹ️ Info message example")

#     st.header("Exception Display")
#     try:
#         1 / 0
#     except Exception as e:
#         st.exception(e)







# Read the original dataset
# def convert_to_csv():
#     dataframe = pd.read_csv('FastagFraudDetection.csv')

#     dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])
#     dataframe["Hour"] = dataframe["Timestamp"].dt.hour
#     dataframe["DayOfWeek"] = dataframe["Timestamp"].dt.dayofweek
#     dataframe["Month"] = dataframe["Timestamp"].dt.month
#     dataframe[['Latitude', 'Longitude']] = dataframe['Geographical_Location'].str.split(',', expand=True).astype(float)
#     dataframe["State_code"] = dataframe["Vehicle_Plate_Number"].str[:2]

#     dataframe['Fraud_indicator'] = dataframe['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})

#     location_fraud_count = dataframe.groupby(['Latitude', 'Longitude'])['Fraud_indicator'].sum().reset_index()
#     location_fraud_count.rename(columns={'Fraud_indicator': 'Fraud_Count'}, inplace=True)
#     location_fraud_count['Total_Transactions'] = dataframe.groupby(['Latitude', 'Longitude'])['Fraud_indicator'].count().values
#     location_fraud_count['Fraud_Rate'] = location_fraud_count['Fraud_Count'] / location_fraud_count['Total_Transactions']
#     high_fraud_locations = location_fraud_count[location_fraud_count['Fraud_Rate'] > 0.2]
#     # print(high_fraud_locations)
#     dataframe['High_Risk_Area'] = dataframe.apply(
#         lambda row: (row['Latitude'], row['Longitude']) in list(zip(high_fraud_locations['Latitude'], high_fraud_locations['Longitude'])),
#         axis=1
#     )


#     # Select required columns
#     selected_columns = [
#         'Vehicle_Type', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions',
#         'Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Fraud_indicator',
#         'Hour', 'DayOfWeek', 'Month', 'Latitude', 'Longitude', 'High_Risk_Area', 'State_code'
#     ]

#     # Create the cleaned dataframe
#     cleaned_df = dataframe[selected_columns]

#     # Save the cleaned data to a new CSV
#     cleaned_df.to_csv('cleaned_data.csv', index=False)

#     print("✅ Cleaned data saved as 'cleaned_data.csv'")