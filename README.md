# ⚡ End-to-End Time Series Energy Forecasting & MLOps Pipeline

An enterprise-grade, end-to-end machine learning and MLOps pipeline designed to forecast future energy consumption patterns using classical statistical models, modern forecasting frameworks, deep learning, and workflow automation.

## 🚀 Project Overview
Accurate energy forecasting is crucial for grid management and resource allocation. This project processes historical energy consumption data (`AEP_MW`), performs rigorous stationarity tests, implements multiple predictive models, and automates daily forecasting using **Prefect**.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Statistical Modeling:** Statsmodels (SARIMA, ADF Test, Seasonal Decomposition)
- **Modern Forecasting:** Facebook Prophet (with holiday and trend components)
- **Deep Learning:** PyTorch (LSTM Neural Network with Sequence Windowing)
- **MLOps & Automation:** Prefect (Data pipeline orchestration & daily scheduling)
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
ts_forecasting/
├── data/                  # Raw energy consumption dataset
├── forecasts/             # Generated daily forecast CSV outputs
├── notebooks/             # Saved model performance & decomposition charts
├── pipeline/              # Prefect automated pipeline scripts (`schedule.py`)
├── src/                   # Core implementation scripts
│   ├── load_data.py       # Data loading and preprocessing
│   ├── decomposition.py   # Trend, seasonality analysis & ADF test
│   ├── sarima_model.py    # SARIMA baseline model
│   ├── prophet_model.py   # Facebook Prophet model
│   ├── lstm_model.py      # PyTorch LSTM deep learning model
│   └── evaluate.py        # Model evaluation & train-test split
├── requirements.txt       # Project dependencies
└── README.md
