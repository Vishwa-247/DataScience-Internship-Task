**Assignment – Data Science (End-to-End Time Series Forecasting System with API) Objective:** 

Build a **production-ready forecasting system** that: 

1\. Trains **multiple forecasting algorithms** 

2\. Compares and selects the **best model** 

3\. Exposes predictions via a **REST API** 

4\. Should be designed like real backend service 

**Dataset:** 

Use the attached data set for your case study 

Shared in [excel file](https://docs.google.com/spreadsheets/d/1I1sFHSOZa9tdQfCahF1W71L4hPrJxkf1/edit?gid=1562810345#gid=1562810345) 

**Problem Statement** 

Forecast **next 8weeks of sales** for each state using historical data. Your solution must: 

• Handle missing dates / missing values (if any) 

• Handle seasonality & trend 

• Automatically select the best performing model 

• Serve predictions via API 

**Mandatory Models to Implement** 

Train and compare at least: 

1\. **ARIMA** / **SARIMA** 

2\. **Facebook Prophet** 

3\. **XGBoost** (with lag features) 

4\. **LSTM** (deep learning) 

**Feature Engineering (critical part)** 

You must create: 

• Lag features (t-1, t-7, t-30) 

• Rolling mean / std 

• Day of week, month, holiday flag 

• Train / validation split using time series logic (no leakage) 

Create a short video of your solution and share it along with Code and documentation