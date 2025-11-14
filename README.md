# 🦠 COVID-19 Historical Case Prediction API

## 🎯 Problem Description
In any pandemic situation, government public health agencies need reliable short-term forecasts of covid-19 disease cases to allocate resources. However, forecasting into the unknown future is unverifiable.  

This project **predicts daily new COVID-19 cases for historical dates (up to August 2024)** using only data available before that date, as the data is no longer kept-up-date.
Because actual values are known, we can **validate model accuracy transparently** — enabling honest assessment of predictive performance.  

The solution is a REST API that returns:
- Predicted case count based on given date prior to August 2024
- Actual reported value (for verification)
- Confidence in trend (via binary classification)

Use cases:  for any prediction of infectious disease given enough data, for public health education

---

## 📥 Data
We use the [Our World in Data (OWID) COVID-19 dataset](https://github.com/owid/covid-19-data), last updated August 2024.  The raw dataset is in https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv


✅ **No need to commit large CSV** — data is downloaded automatically during training.

To download manually:
```bash
curl -L "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv" -o data/owid-covid-data.csv
```
---

## 🧪 How to Run
### Option 1: Local(with virtual environment on Windows)

``` bash
# Create and activate environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run API
python predict.py
```

Visit: http://localhost:8000/docs


### Option 2:Docker
``` bash
docker build -t covid-api .
docker run -p 8000:8000 covid-api
```

### Option 3: Cloud (Render - free site for small projects)
See deployment below:


🔍 Notebook
Explore notebook.ipynb for:

- Data cleaning
- EDA (distributions, missing values, time trends)
- Feature importance (SHAP/XGBoost)
- Model comparison & hyperparameter tuning

🚀 Deployment
Deployed live on Render (free tier):
👉 https://covid-forecast-api.onrender.com

Test it:
``` bash
curl -X POST "https://covid-forecast-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"country": "Australia", "date": "2024-07-15"}'
```
