import streamlit as st
import pandas as pd
import numpy as np
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TEST_SIZE    = 6
FORECAST_HORIZON = 12
N_LAGS       = 6
FEATURE_COLS_BASE = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + [
    "rolling_mean_3", "month_num", "year"
]

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["OrderDate"])

df = load()

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Global Filters")

date_input = st.sidebar.date_input(
    "Date Range",
    value=[df["OrderDate"].min().date(), df["OrderDate"].max().date()],
)
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_date, end_date = date_input
else:
    start_date = end_date = date_input

fdf = df[df["OrderDate"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]

if fdf.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# AGGREGATE → MONTHLY REVENUE
# ─────────────────────────────────────────────────────────────

monthly_revenue = (
    fdf.set_index("OrderDate")["TotalAmount"]
    .resample("MS")
    .sum()
    .sort_index()
)

# Fill any gaps so the series is always contiguous
full_range = pd.date_range(
    monthly_revenue.index.min(), monthly_revenue.index.max(), freq="MS"
)
monthly_revenue = monthly_revenue.reindex(full_range, fill_value=0)
monthly_revenue.name = "Revenue"

MIN_MONTHS = N_LAGS + 6 + TEST_SIZE   # lags + rolling window safety + test
if len(monthly_revenue) < MIN_MONTHS:
    st.error(
        f"Not enough monthly data ({len(monthly_revenue)} months). "
        f"Need at least {MIN_MONTHS} months. Try widening the date range."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def build_features(series):
    df_feat = pd.DataFrame({"Month": series.index, "Revenue": series.values})
    for lag in range(1, N_LAGS + 1):
        df_feat[f"lag_{lag}"] = df_feat["Revenue"].shift(lag)
    df_feat["rolling_mean_3"] = df_feat["Revenue"].shift(1).rolling(3).mean()
    df_feat["month_num"] = df_feat["Month"].dt.month
    df_feat["year"]      = df_feat["Month"].dt.year
    return df_feat.dropna().reset_index(drop=True)


def compute_metrics(name, y_true, y_pred):
    mask = y_true != 0
    mape = (
        round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, 2)
        if mask.any() else float("nan")
    )
    return {
        "Model":    name,
        "MAE":      round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R²":       round(r2_score(y_true, y_pred), 4),
        "MAPE (%)": mape,
    }


def xgb_recursive_forecast(model,history,horizon,):
    known = list(history.values)
    last_date = history.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                  periods=horizon, freq="MS")
    preds = []

    for step, future_date in enumerate(future_dates):
        lags = [known[-(i)] for i in range(1, N_LAGS + 1)]
        rolling_mean_3 = np.mean(known[-3:])
        month_num = future_date.month
        year      = future_date.year

        row = np.array([*lags, rolling_mean_3, month_num, year], dtype=float).reshape(1, -1)
        pred = float(model.predict(row)[0])
        pred = max(pred, 0)
        preds.append(pred)
        known.append(pred)

    return future_dates, np.array(preds)

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────────────────────

df_ts = build_features(monthly_revenue)

with st.expander("Lagged Dataset Preview"):
    st.dataframe(df_ts.head(10), use_container_width=True)

train = df_ts.iloc[:-TEST_SIZE].copy()
test  = df_ts.iloc[-TEST_SIZE:].copy()

X_train, y_train = train[FEATURE_COLS_BASE], train["Revenue"]
X_test,  y_test  = test[FEATURE_COLS_BASE],  test["Revenue"]

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING  (cached so rerun doesn't refit)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def train_models(
    X_tr, y_tr,prophet_train_df):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr)

    # Prophet
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    prophet_model.fit(prophet_train_df)

    return lr, xgb, prophet_model


prophet_train_end = test["Month"].iloc[0]
prophet_train_df  = pd.DataFrame({
    "ds": monthly_revenue.index,
    "y":  monthly_revenue.values,
})[lambda d: d["ds"] < prophet_train_end]

lr, xgb, prophet_model = train_models(
    X_train, y_train, prophet_train_df
)

# ─────────────────────────────────────────────────────────────
# TEST-SET PREDICTIONS
# ─────────────────────────────────────────────────────────────

y_true     = y_test.values
naive_pred = test["lag_1"].values
lr_pred    = lr.predict(X_test)
xgb_pred   = xgb.predict(X_test)

future_prophet  = prophet_model.make_future_dataframe(periods=TEST_SIZE, freq="MS")
prophet_pred = (
    prophet_model.predict(future_prophet)
    .set_index("ds")["yhat"]
    .reindex(test["Month"].values)
    .clip(lower=0)
    .values
)

# ─────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────

results = pd.DataFrame([
    compute_metrics("Naive",             y_true, naive_pred),
    compute_metrics("Linear Regression", y_true, lr_pred),
    compute_metrics("XGBoost",           y_true, xgb_pred),
    compute_metrics("Prophet",           y_true, prophet_pred),
]).set_index("Model")

# ─────────────────────────────────────────────────────────────
# UI — MODEL SELECTOR & ACTUAL vs PREDICTED
# ─────────────────────────────────────────────────────────────

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

chart_df = pd.DataFrame({
    "Month":     test["Month"].values,
    "Actual":    y_true,
    "Predicted": pred_map[model_choice],
})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=chart_df["Month"], y=chart_df["Actual"],
    name="Actual", line=dict(color="#1f2937", width=2), mode="lines+markers",
))
fig.add_trace(go.Scatter(
    x=chart_df["Month"], y=chart_df["Predicted"],
    name="Predicted", line=dict(color="#f59e0b", width=2, dash="dot"),
    mode="lines+markers",
))
fig.update_layout(
    title=f"{model_choice} — Actual vs Predicted Revenue",
    template="plotly_white", hovermode="x unified",
    xaxis_title="Month", yaxis_title="Revenue ($)", height=380,
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# UI — ALL MODELS COMPARISON
# ─────────────────────────────────────────────────────────────

with st.expander("Compare all models side by side"):
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=test["Month"], y=y_true,
        name="Actual", line=dict(color="#1f2937", width=2.5),
    ))
    colors = {
        "Naive":              "#9ca3af",
        "Linear Regression":  "#3b82f6",
        "XGBoost":            "#f59e0b",
        "Prophet":            "#ef4444",
    }
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
# UI — METRICS TABLE
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
# FUTURE FORECAST — XGBoost (recursive one-step-ahead)
# ─────────────────────────────────────────────────────────────

st.subheader(f"Future Forecast (XGBoost — next {FORECAST_HORIZON} months)")

# Use the FULL monthly_revenue series as history so all lags are well-defined
future_dates, xgb_future_pred = xgb_recursive_forecast(
    model=xgb,
    history=monthly_revenue,
    horizon=FORECAST_HORIZON,
)

fig2 = go.Figure()

# Historical actuals
fig2.add_trace(go.Scatter(
    x=monthly_revenue.index, y=monthly_revenue.values,
    name="Historical", line=dict(color="#1f2937", width=2),
))

# XGBoost forecast
fig2.add_trace(go.Scatter(
    x=future_dates, y=xgb_future_pred,
    name="XGBoost Forecast",
    line=dict(color="#f59e0b", width=2.5, dash="dot"),
    mode="lines+markers", marker=dict(size=7, symbol="circle"),
))

# Thin vertical line separating history from forecast
forecast_start = monthly_revenue.index[-1].isoformat()

fig2.add_shape(
    type="line",
    x0=forecast_start, x1=forecast_start,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(width=1, dash="dash", color="#6b7280"),
)
fig2.add_annotation(
    x=forecast_start,
    y=1,
    xref="x", yref="paper",
    text="Forecast start",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(color="#6b7280", size=12),
)


fig2.update_layout(
    title=f"XGBoost — Next {FORECAST_HORIZON} Months Revenue Forecast (Recursive)",
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Revenue ($)",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig2, use_container_width=True)

# Download forecast
forecast_df = pd.DataFrame({
    "Month":    future_dates.strftime("%Y-%m-%d"),
    "Forecast": xgb_future_pred.round(2),
})
st.download_button(
    "⬇ Download XGBoost forecast CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="xgboost_forecast.csv",
    mime="text/csv",
)