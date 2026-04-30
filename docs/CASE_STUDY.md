# Assignment — Data Science Internship

**End-to-End Time Series Forecasting System with API**

## Objective

Build a **production-ready forecasting system** that:

1. Trains **multiple forecasting algorithms**
2. Compares and selects the **best model**
3. Exposes predictions via a **REST API**
4. Should be designed like a real backend service

## Dataset

- **Source:** Attached Excel file (weekly US beverage sales)
- **Link:** [Google Sheets](https://docs.google.com/spreadsheets/d/1I1sFHSOZa9tdQfCahF1W71L4hPrJxkf1/edit?gid=1562810345#gid=1562810345)

## Problem Statement

Forecast the **next 8 weeks of sales** for each US state using historical data. The solution must:

- Handle missing dates / missing values
- Handle seasonality & trend
- Automatically select the best performing model
- Serve predictions via API

## Mandatory Models

Train and compare at least:

1. **ARIMA / SARIMA**
2. **Facebook Prophet**
3. **XGBoost** (with lag features)
4. **LSTM** (deep learning)

## Feature Engineering (critical)

Must create:

- Lag features (t-1, t-7, t-30)
- Rolling mean / std
- Day of week, month, holiday flag
- Train / validation split using time series logic (no leakage)

## Deliverables

Create a short video of your solution and share it along with code and documentation.
