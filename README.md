# 🚗 Fastag Fraud Detection Using Machine Learning

This project aims to detect fraudulent Fastag transactions using machine learning techniques. It includes data preprocessing, model training, and a prediction system built with Python.

---

## 📁 Project Structure

Major Project/
├── app.py # Main app script
├── model.py # Model building code
├── prediction.py # Inference script
├── live_stats.py # Real-time statistics
├── model_features.joblib # Serialized model features
├── random_forest_model.joblib # Trained Random Forest model
├── label_encoders.joblib # Encoders for categorical data
├── data/ # Raw and cleaned datasets
│ ├── cleaned_data.csv
│ ├── FastagFraudDetection.csv
│ └── ...
├── notebooks/ # Jupyter notebooks for EDA & modeling
│ └── Fastag Fraud detection.ipynb
├── Documentation/ # Abstracts and reports
│ └── Project documentation.pdf
└── venv/ # Python virtual environment (optional)

yaml
Copy
Edit

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
Note: Create a requirements.txt using:
pip freeze > requirements.txt (excluding virtual environment packages)

Usage
Run the main app:

bash
Copy
Edit
python app.py
Train or update model (optional):

bash
Copy
Edit
python model.py
Make predictions:

bash
Copy
Edit
python prediction.py
🧠 ML Approach
Model Used: Random Forest Classifier

Features: Transaction attributes, time-based features, encoded categorical variables

Tools: Pandas, Scikit-learn, Joblib

📄 Documentation
All documentation including abstract, project report, and presentation slides are available in the Documentation/ folder.

📊 Datasets
Datasets used in this project are located in the data/ folder and include:

FastagFraudDetection.csv

netc_yearly_data.csv

NETC_Monthly_Transactions.csv

✅ Future Improvements
Web dashboard integration

Real-time fraud alerts

Model explainability (e.g., SHAP)
