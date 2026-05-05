import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prophet import Prophet
from pathlib import Path

st.title("Revenue Forecasting System")
st.caption("Multi-model time series forecasting & comparison")
st.divider()



# ── Load data ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # goes up from app/ to root
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

@st.cache_data
def load():
    df = pd.read_csv(DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["OrderDate"])
    return df

df = load()

# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.header("Global Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=[df["OrderDate"].min(), df["OrderDate"].max()]
)
categories = st.sidebar.multiselect(
    "Categories", df["Category"].unique(), default=df["Category"].unique()
)
countries = st.sidebar.multiselect(
    "Countries", df["Country"].unique(), default=df["Country"].unique()
)

mask = (
    df["OrderDate"].between(
        pd.Timestamp(date_range[0]),
        pd.Timestamp(date_range[1])
    ) &
    df["Category"].isin(categories) &
    df["Country"].isin(countries)
)
fdf = df[mask]

# ─────────────────────────────────────────────────────────────
# LOAD DATA 
# ─────────────────────────────────────────────────────────────

df_ts = (
    fdf.set_index("OrderDate")
       .resample("ME")["TotalAmount"]
       .sum()
       .reset_index()
)

df_ts.columns = ["Month", "Revenue"]

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (for ML models)
# ─────────────────────────────────────────────────────────────

for lag in range(1, 7):
    df_ts[f"lag_{lag}"] = df_ts["Revenue"].shift(lag)

df_ts["rolling_mean_3"] = df_ts["Revenue"].rolling(3).mean()
df_ts = df_ts.dropna()

# time features
df_ts["month_num"] = df_ts["Month"].dt.month
df_ts["year"] = df_ts["Month"].dt.year

# ─────────────────────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────────────────────

train = df_ts.iloc[:-6]
test = df_ts.iloc[-6:]

X_train = train.drop(["Month", "Revenue"], axis=1)
y_train = train["Revenue"]

X_test = test.drop(["Month", "Revenue"], axis=1)
y_test = test["Revenue"]

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

# 1. Naive baseline
naive_pred = test["lag_1"]

# 2. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 3. XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# ─────────────────────────────────────────────────────────────
# PROPHET MODEL
# ─────────────────────────────────────────────────────────────

prophet_df = df_ts[["Month", "Revenue"]].rename(columns={"Month": "ds", "Revenue": "y"})

prophet_model = Prophet()
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=6, freq="ME")
forecast = prophet_model.predict(future)

prophet_pred = forecast["yhat"].iloc[-6:].values

# ─────────────────────────────────────────────────────────────
# METRICS FUNCTION
# ─────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

results = pd.DataFrame({
    "Model": ["Naive", "Linear", "XGBoost", "Prophet"],
    "MAE": [
        metrics(y_test, naive_pred)["MAE"],
        metrics(y_test, lr_pred)["MAE"],
        metrics(y_test, xgb_pred)["MAE"],
        metrics(y_test, prophet_pred)["MAE"],
    ],
    "RMSE": [
        metrics(y_test, naive_pred)["RMSE"],
        metrics(y_test, lr_pred)["RMSE"],
        metrics(y_test, xgb_pred)["RMSE"],
        metrics(y_test, prophet_pred)["RMSE"],
    ],
    "R2": [
        metrics(y_test, naive_pred)["R2"],
        metrics(y_test, lr_pred)["R2"],
        metrics(y_test, xgb_pred)["R2"],
        metrics(y_test, prophet_pred)["R2"],
    ]
})

# ─────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────

model_choice = st.selectbox(
    "Select Model",
    ["Naive", "Linear Regression", "XGBoost", "Prophet"]
)

# ─────────────────────────────────────────────────────────────
# PREDICTION SELECTION
# ─────────────────────────────────────────────────────────────

if model_choice == "Naive":
    pred = naive_pred
elif model_choice == "Linear Regression":
    pred = lr_pred
elif model_choice == "XGBoost":
    pred = xgb_pred
else:
    pred = prophet_pred

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

chart_df = pd.DataFrame({
    "Month": test["Month"].values,
    "Actual": y_test.values,
    "Predicted": pred
})

fig = px.line(
    chart_df,
    x="Month",
    y=["Actual", "Predicted"],
    title=f"{model_choice} — Actual vs Predicted Revenue",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────

st.subheader("Model Performance Comparison")
st.dataframe(results, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FUTURE FORECAST (Prophet-based)
# ─────────────────────────────────────────────────────────────

st.subheader("Future Forecast (Prophet)")

future_chart = forecast[["ds", "yhat"]].tail(12)

fig2 = px.line(
    future_chart,
    x="ds",
    y="yhat",
    title="Next Months Revenue Forecast",
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)