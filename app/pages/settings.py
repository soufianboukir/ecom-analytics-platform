import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from pathlib import Path

st.title("Revenue Forecasting System")
st.caption("Multi-model time series forecasting & comparison")
st.divider()

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

@st.cache_data
def load():
    df = pd.read_csv(DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["OrderDate"])
    return df

df = load()

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Global Filters")

# FIX 1: guard against single-date selection (returns a single date, not a tuple)
date_input = st.sidebar.date_input(
    "Date Range",
    value=[df["OrderDate"].min().date(), df["OrderDate"].max().date()]
)
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_date, end_date = date_input
else:
    start_date = end_date = date_input

mask = (
    df["OrderDate"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
)
fdf = df[mask]

# FIX 2: stop early if filters return no data
if fdf.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# AGGREGATE TO MONTHLY REVENUE
# ─────────────────────────────────────────────────────────────

# FIX 3: use "MS" (month-start) instead of "ME" — Prophet requires month-start
monthly_revenue = (
    fdf.set_index("OrderDate")["TotalAmount"]
    .resample("MS")
    .sum()
    .sort_index()
)

# FIX 4: fill any missing months so the series is always contiguous
full_range      = pd.date_range(monthly_revenue.index.min(),
                                monthly_revenue.index.max(), freq="MS")
monthly_revenue = monthly_revenue.reindex(full_range, fill_value=0)
monthly_revenue.name = "Revenue"

# Need at least 18 months: 12 for lag features + 6 for test
if len(monthly_revenue) < 18:
    st.error(
        f"Not enough monthly data ({len(monthly_revenue)} months). "
        "Need at least 18 months. Try widening the date range or removing filters."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (ML models only — no leakage)
# ─────────────────────────────────────────────────────────────

df_ts = pd.DataFrame({"Month": monthly_revenue.index, "Revenue": monthly_revenue.values})

for lag in range(1, 7):
    df_ts[f"lag_{lag}"] = df_ts["Revenue"].shift(lag)

# FIX 5: shift before rolling to avoid using the current row's own value
df_ts["rolling_mean_3"] = df_ts["Revenue"].shift(1).rolling(3).mean()

df_ts["month_num"] = df_ts["Month"].dt.month
df_ts["year"]      = df_ts["Month"].dt.year

df_ts = df_ts.dropna().reset_index(drop=True)


st.subheader("Lagged Dataset Preview")
st.dataframe(df_ts.head(10))

# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT  (time-based, no shuffling)
# ─────────────────────────────────────────────────────────────

TEST_SIZE = 6

train = df_ts.iloc[:-TEST_SIZE].copy()
test  = df_ts.iloc[-TEST_SIZE:].copy()

FEATURE_COLS = [c for c in df_ts.columns if c not in ("Month", "Revenue")]

X_train, y_train = train[FEATURE_COLS], train["Revenue"]
X_test,  y_test  = test[FEATURE_COLS],  test["Revenue"]

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

# 1. Naive baseline — previous month's actual revenue
# FIX 6: use .values to avoid index misalignment in metrics
naive_pred = test["lag_1"].values

# 2. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 3. XGBoost
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# ─────────────────────────────────────────────────────────────
# PROPHET
# FIX 7: use the full monthly_revenue series (before NaN-drop from lag
#         engineering) so Prophet sees all months including the early ones
# ─────────────────────────────────────────────────────────────

prophet_train_end = test["Month"].iloc[0]  # first test month

prophet_df = pd.DataFrame({
    "ds": monthly_revenue.index,
    "y":  monthly_revenue.values,
})
prophet_train_df = prophet_df[prophet_df["ds"] < prophet_train_end]

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
)
prophet_model.fit(prophet_train_df)

# FIX 8: Prophet make_future_dataframe needs freq="MS" to match our index
future   = prophet_model.make_future_dataframe(periods=TEST_SIZE, freq="MS")
forecast = prophet_model.predict(future)

# Align Prophet predictions to the exact test dates
test_months  = test["Month"].values
prophet_pred = (
    forecast.set_index("ds")["yhat"]
    .reindex(test_months)
    .clip(lower=0)          # revenue cannot be negative
    .values
)

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero for any month with zero revenue
    mask = y_true != 0
    return round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, 2)

def compute_metrics(name: str, y_true, y_pred) -> dict:
    """Return MAE, RMSE, R², and MAPE for one model."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "Model": name,
        "MAE":      round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R²":       round(r2_score(y_true, y_pred), 4),
        "MAPE (%)": mape(y_true, y_pred),
    }

y_true = y_test.values

results = pd.DataFrame([
    compute_metrics("Naive",             y_true, naive_pred),
    compute_metrics("Linear Regression", y_true, lr_pred),
    compute_metrics("XGBoost",           y_true, xgb_pred),
    compute_metrics("Prophet",           y_true, prophet_pred),
]).set_index("Model")

# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────

# Model selector
model_choice = st.selectbox(
    "Select model to inspect",
    ["Naive", "Linear Regression", "XGBoost", "Prophet"],
)

pred_map = {
    "Naive":              naive_pred,
    "Linear Regression":  lr_pred,
    "XGBoost":            xgb_pred,
    "Prophet":            prophet_pred,
}
selected_pred = pred_map[model_choice]

# ─────────────────────────────────────────────────────────────
# CHART 1 — Actual vs Predicted (selected model)
# ─────────────────────────────────────────────────────────────

chart_df = pd.DataFrame({
    "Month":     test["Month"].values,
    "Actual":    y_true,
    "Predicted": selected_pred,
})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=chart_df["Month"], y=chart_df["Actual"],
    name="Actual", line=dict(color="#1f2937", width=2), mode="lines+markers"
))
fig.add_trace(go.Scatter(
    x=chart_df["Month"], y=chart_df["Predicted"],
    name="Predicted", line=dict(color="#f59e0b", width=2, dash="dot"),
    mode="lines+markers"
))
fig.update_layout(
    title=f"{model_choice} — Actual vs Predicted Revenue",
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Revenue ($)",
    height=380,
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# CHART 2 — All models on one chart (for comparison)
# ─────────────────────────────────────────────────────────────

with st.expander("Compare all models side by side"):
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=test["Month"], y=y_true,
        name="Actual", line=dict(color="#1f2937", width=2.5)
    ))
    colors = {"Naive": "#9ca3af", "Linear Regression": "#3b82f6",
              "XGBoost": "#f59e0b", "Prophet": "#ef4444"}
    for name, pred in pred_map.items():
        fig_all.add_trace(go.Scatter(
            x=test["Month"], y=pred, name=name,
            line=dict(color=colors[name], width=1.8, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
        ))
    fig_all.update_layout(
        title="All Models — Actual vs Predicted",
        template="plotly_white", hovermode="x unified",
        xaxis_title="Month", yaxis_title="Revenue ($)", height=380,
    )
    st.plotly_chart(fig_all, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────

st.subheader("Model Performance Comparison")
st.dataframe(
    results.style
        .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#1ec26d")
        .highlight_max(subset=["R²"],                       color="#33b371")
        .format({
            "MAE":      "{:,.2f}",
            "RMSE":     "{:,.2f}",
            "R²":       "{:.4f}",
            "MAPE (%)": "{:.2f}%",
        }),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────
# FUTURE FORECAST — Prophet
# ─────────────────────────────────────────────────────────────

st.subheader("Future Forecast (Prophet — next 12 months)")

future_extended  = prophet_model.make_future_dataframe(
    periods=TEST_SIZE + 12, freq="MS"
)
forecast_extended = prophet_model.predict(future_extended)
future_only = forecast_extended[
    forecast_extended["ds"] > monthly_revenue.index[-1]
][["ds", "yhat", "yhat_lower", "yhat_upper"]].head(12)

fig2 = go.Figure()

# Historical actuals
fig2.add_trace(go.Scatter(
    x=monthly_revenue.index, y=monthly_revenue.values,
    name="Historical", line=dict(color="#1f2937", width=2)
))

# Confidence band
fig2.add_trace(go.Scatter(
    x=pd.concat([future_only["ds"], future_only["ds"].iloc[::-1]]),
    y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"].iloc[::-1]]),
    fill="toself", fillcolor="rgba(239,68,68,0.1)",
    line=dict(color="rgba(0,0,0,0)"), name="80% Confidence Interval",
))

# Forecast line
fig2.add_trace(go.Scatter(
    x=future_only["ds"], y=future_only["yhat"].clip(lower=0),
    name="Prophet Forecast",
    line=dict(color="#ef4444", width=2, dash="dot"),
    mode="lines+markers", marker=dict(size=6),
))

fig2.update_layout(
    title="Prophet — Next 12 Months Revenue Forecast",
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Revenue ($)",
    height=400,
)
st.plotly_chart(fig2, use_container_width=True)

# Download forecast
st.download_button(
    "Download forecast CSV",
    data=future_only.rename(columns={
        "ds": "Month", "yhat": "Forecast",
        "yhat_lower": "Lower bound", "yhat_upper": "Upper bound"
    }).to_csv(index=False).encode("utf-8"),
    file_name="prophet_forecast.csv",
    mime="text/csv",
)