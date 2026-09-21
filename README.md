The interesting part is that the project will have two data flows:

                 ┌─────────────────────┐
                 │   Open-Meteo API    │
                 │   Live Weather Data │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Data Collection   │
                 │      Python         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Cleaning &     │
                 │ Feature Engineering │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ ML Model Training   │
                 │ Logistic Regression │
                 │     / XGBoost       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Saved Model (.pkl)  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Streamlit       │
                 │    Web Dashboard    │
                 └─────────────────────┘

Open-Meteo is particularly suitable because its weather API provides JSON data without requiring an API key, and it provides both forecast and historical weather data.

1. What will our project predict?
Target

We'll predict:

Will it rain?

So this is a binary classification problem:

0 → No Rain
1 → Rain
Features

For example:

temperature
relative_humidity
precipitation
wind_speed
cloud_cover
pressure

The model learns relationships between these features and the rain/no-rain target.

2. Why this is a good ML project

This project lets you demonstrate the complete ML lifecycle:

API
 ↓
Data Collection
 ↓
Data Validation
 ↓
EDA
 ↓
Data Cleaning
 ↓
Feature Engineering
 ↓
Train/Test Split
 ↓
Preprocessing
 ↓
Logistic Regression
 ↓
XGBoost
 ↓
Model Evaluation
 ↓
Model Selection
 ↓
Model Serialization
 ↓
FastAPI/Backend layer
 ↓
Streamlit
 ↓
Live Prediction

This is much closer to a real Data Scientist / ML Engineer project than simply training a model on a CSV.

3. Complete folder structure

I recommend this structure:

live-weather-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── data/
│   ├── raw/
│   │   └── weather_raw.csv
│   │
│   ├── processed/
│   │   └── weather_processed.csv
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   │
│   ├── __init__.py
│   │
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── api_client.py
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_logistic.py
│   │   ├── train_xgboost.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py
│
├── models/
│   ├── logistic_model.pkl
│   ├── xgboost_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── reports/
│   ├── figures/
│   │   ├── rainfall_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   │
│   └── model_report.txt
│
├── streamlit_app/
│   ├── app.py
│   │
│   ├── pages/
│   │   ├── 1_📊_EDA.py
│   │   ├── 2_🤖_Prediction.py
│   │   ├── 3_📈_Model_Performance.py
│   │   └── 4_🌦️_Live_Weather.py
│   │
│   └── components/
│       ├── charts.py
│       └── ui.py
│
└── tests/
    ├── test_api.py
    ├── test_preprocessing.py
    └── test_model.py

This gives you a production-style project structure.

4. Live API

We'll use:

Open-Meteo Weather API

It supports current/hourly weather variables such as temperature, humidity, precipitation, cloud cover and wind speed.

It also provides historical weather data, including hourly temperature, humidity, precipitation and wind variables.

For example:

Location:
Delhi

Latitude:
28.6139

Longitude:
77.2090

The API request conceptually looks like:

https://api.open-meteo.com/v1/forecast

with parameters such as:

latitude
longitude
hourly
temperature_2m
relative_humidity_2m
precipitation
cloud_cover
wind_speed_10m
5. Important ML point

There is one issue we need to handle correctly.

The live API gives today's/future weather, but our model needs historical examples containing a target.

So we'll create the training dataset using historical weather data.

For example:

Date        Temp   Humidity   Cloud   Wind   Rain
---------------------------------------------------
2025-01-01   18      65        20      8      0
2025-01-02   17      82        75      12     1
2025-01-03   19      70        40      10     0
...

Then:

X = weather features
Y = rain/no-rain

The historical API can provide weather records going back many years, which gives us enough observations for training.

6. Data collection architecture

Create:

src/data/api_client.py

Its job:

API
 ↓
HTTP Request
 ↓
JSON Response
 ↓
Pandas DataFrame

Example:

import requests
import pandas as pd


def get_weather_data(
    latitude,
    longitude,
    start_date,
    end_date
):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    return df
7. Dataset

Our raw data might look like:

time
temperature_2m
relative_humidity_2m
precipitation
cloud_cover
wind_speed_10m

Then we create:

rain

For example:

df["rain"] = (df["precipitation"] > 0).astype(int)

So:

precipitation = 0
        ↓
rain = 0


precipitation > 0
        ↓
rain = 1
8. Feature engineering

This is where your project becomes more interesting.

We'll create features such as:

temperature
humidity
cloud_cover
wind_speed
pressure
previous_hour_rain
rolling_rain
hour
day
month

For example:

df["hour"] = df["time"].dt.hour

df["day"] = df["time"].dt.day

df["month"] = df["time"].dt.month

And lag features:

df["previous_hour_rain"] = (
    df["precipitation"]
    .shift(1)
)

This allows the model to use previous weather conditions.

9. EDA

Notebook:

02_eda.ipynb

We'll investigate:

Distribution
Temperature
Humidity
Wind
Precipitation
Relationships

For example:

Humidity vs Rain
Cloud Cover vs Rain
Temperature vs Rain
Wind vs Rain
Correlation matrix
                 Rain
Temperature     -0.20
Humidity         0.65
Cloud Cover      0.58
Wind             0.12

These numbers are just illustrative; we'll calculate the actual values from the API data.

10. Preprocessing pipeline

We'll use:

Raw Data
   ↓
Missing Values
   ↓
Feature Selection
   ↓
Scaling
   ↓
Model

For Logistic Regression:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
11. Logistic Regression

This becomes our baseline model.

Weather features
       ↓
Logistic Regression
       ↓
Probability
       ↓
Rain / No Rain

Example:

Probability of Rain = 0.83

Then:

0.83 > 0.50

Prediction = Rain
12. XGBoost

Then we'll train:

XGBoost Classifier

Example:

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

We'll tune these parameters rather than blindly choosing them.

13. Model comparison

We'll compare:

Metric	Logistic Regression	XGBoost
Accuracy	calculated	calculated
Precision	calculated	calculated
Recall	calculated	calculated
F1	calculated	calculated
ROC-AUC	calculated	calculated

We shouldn't choose a model simply because it has the highest accuracy.

For rain prediction, class imbalance can make accuracy misleading.

We'll therefore examine:

Confusion Matrix
Precision
Recall
F1
ROC-AUC
PR-AUC
14. Streamlit application

Your Streamlit application will have:

Weather ML Predictor
│
├── Dashboard
│
├── Live Weather
│
├── Prediction
│
├── EDA
│
└── Model Performance
Dashboard

Something like:

╔══════════════════════════════════════╗
║       🌦️ WEATHER ML PREDICTOR        ║
╠══════════════════════════════════════╣
║                                      ║
║  Temperature       28.4 °C            ║
║  Humidity          76 %               ║
║  Wind Speed        11 km/h            ║
║  Cloud Cover       82 %               ║
║                                      ║
║  ────────────────────────────────     ║
║                                      ║
║       🌧️ RAIN PREDICTION             ║
║                                      ║
║       Probability: 81.7%              ║
║       Prediction: RAIN                ║
║                                      ║
╚══════════════════════════════════════╝
15. Live prediction flow

This is the most important part.

User opens Streamlit:

Select City
     ↓
Get Latitude/Longitude
     ↓
Call Weather API
     ↓
Get Current Weather
     ↓
Preprocess Features
     ↓
Load Trained Model
     ↓
Prediction
     ↓
Display Result

So the application isn't using a static CSV for prediction.

It's:

LIVE API
   ↓
LIVE WEATHER
   ↓
ML MODEL
   ↓
LIVE PREDICTION
16. Streamlit prediction page

For example:

import streamlit as st
import joblib

model = joblib.load(
    "models/xgboost_model.pkl"
)

st.title("🌦️ Weather Rain Prediction")

temperature = st.number_input(
    "Temperature"
)

humidity = st.number_input(
    "Humidity"
)

cloud_cover = st.number_input(
    "Cloud Cover"
)

wind_speed = st.number_input(
    "Wind Speed"
)

if st.button("Predict"):

    data = [[
        temperature,
        humidity,
        cloud_cover,
        wind_speed
    ]]

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(
            f"🌧️ Rain Expected — {probability:.2%}"
        )
    else:
        st.success(
            f"☀️ No Rain Expected — "
            f"{1-probability:.2%}"
        )

Later we'll replace those manually entered values with live API values.

17. Better architecture

I would actually make the final architecture:

                    USER
                     │
                     ▼
              ┌─────────────┐
              │  Streamlit  │
              └──────┬──────┘
                     │
          ┌──────────┴───────────┐
          │                      │
          ▼                      ▼
   Live Weather API        ML Prediction
          │                      │
          │                ┌─────┴─────┐
          │                │           │
          │             Logistic    XGBoost
          │                │           │
          └────────────────┴───────────┘
                           │
                           ▼
                    Prediction Result
18. Requirements

Create:

requirements.txt

with:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
requests
streamlit
joblib
python-dotenv
19. .gitignore
__pycache__/
*.pyc

.venv/
venv/
env/

.env

.ipynb_checkpoints/

data/raw/*
models/*.pkl

.DS_Store

We don't want to accidentally commit secrets or unnecessary generated files.

20. README structure

Your GitHub README should contain:

# 🌦️ Live Weather ML Prediction System

## Project Overview

## Business Problem

## Objective

## Dataset

## Live API

## Architecture

## Data Pipeline

## Exploratory Data Analysis

## Feature Engineering

## Machine Learning Models

### Logistic Regression

### XGBoost

## Model Evaluation

## Streamlit Application

## Project Structure

## Installation

## How to Run

## API Integration

## Results

## Future Improvements
21. How you'll run it

From the project root:

cd live-weather-ml

Create environment:

python -m venv .venv

Activate:

.venv\Scripts\activate

Install:

pip install -r requirements.txt

Then collect data:

python -m src.data.data_loader

Train:

python -m src.models.train_logistic

and:

python -m src.models.train_xgboost

Finally:

streamlit run streamlit_app/app.py
22. Project development phases

I suggest we build this from scratch, not dump all the code at once.

Phase 1 — Project setup
Folder structure
Virtual environment
requirements.txt
Git
README
Phase 2 — API
Open-Meteo
API request
JSON
DataFrame
Save raw data
Phase 3 — Data engineering
Cleaning
Missing values
Duplicates
Data types
Feature engineering
Phase 4 — EDA
Univariate analysis
Bivariate analysis
Correlation
Outliers
Class imbalance
Phase 5 — ML
Train/test split
Baseline
Logistic Regression
XGBoost
Hyperparameter tuning
Cross-validation
Phase 6 — Evaluation
Confusion Matrix
Precision
Recall
F1
ROC-AUC
PR-AUC
Feature importance
Phase 7 — Deployment
joblib
model.pkl
Streamlit
Live API
Live prediction
Phase 8 — Production improvements
Logging
Error handling
API timeout handling
Caching
Model versioning
Testing
Docker
GitHub
Deployment
One important improvement I'd make

Instead of making this merely "Rain Prediction", we can turn it into a proper project called:

🌦️ Live Weather Intelligence & Rain Prediction System

And eventually add:

             Live Weather
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Temperature  Humidity   Wind
       │          │          │
       └──────────┼──────────┘
                  ▼
            ML Pipeline
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
Logistic Regression       XGBoost
       │                     │
       └──────────┬──────────┘
                  ▼
            Rain Probability
                  │
                  ▼
             Streamlit
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Prediction   Charts    Live Data

This gives you a project where you can explain API integration + data engineering + EDA + feature engineering + classical ML + XGBoost + model evaluation + Streamlit deployment in an interview.
