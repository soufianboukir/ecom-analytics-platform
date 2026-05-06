import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.title("Customer Insights")
st.caption("Lifetime value · Geographic distribution · Payment behaviour")
st.divider()

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

@st.cache_data
def load():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    df["orderdate"] = pd.to_datetime(df["orderdate"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["orderdate"])

df = load()

# ─────────────────────────────────────────────────────────────
# FILTER DATA (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def filter_data(df, start, end):
    return df[df["orderdate"].between(start, end)]

st.sidebar.header("Global Filters")

date_input = st.sidebar.date_input(
    "Date Range",
    value=[df["orderdate"].min().date(), df["orderdate"].max().date()],
)

if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_date, end_date = date_input
else:
    start_date = end_date = date_input

fdf = filter_data(df, pd.Timestamp(start_date), pd.Timestamp(end_date))

if fdf.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────

cust_col  = "customername" if "customername" in fdf.columns else None
city_col  = "city" if "city" in fdf.columns else None
state_col = "state" if "state" in fdf.columns else None
pay_col   = "paymentmethod" if "paymentmethod" in fdf.columns else None

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────

AMBER = "#32a3ff"
DARK  = "#1f2937"
SLATE = "#6b7280"

LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
)


# ─────────────────────────────────────────────────────────────
# LTV — TOP CUSTOMERS
# ─────────────────────────────────────────────────────────────

st.subheader("Top Customers by Lifetime Value")

if cust_col:
    ltv = (
        fdf.groupby(cust_col)["totalamount"]
        .sum()
        .nlargest(20)
        .sort_values()
        .reset_index()
    )

    fig = go.Figure(go.Bar(
        y=ltv[cust_col],
        x=ltv["totalamount"],
        orientation="h",
        
        # Gradient coloring based on value
        marker=dict(
            color=ltv["totalamount"],
            colorscale="Blues",
            showscale=False  # remove side color bar
        ),

        text=ltv["totalamount"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside"
    ))

    fig.update_layout(**LAYOUT, height=500)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Customer column not found")

st.divider()

# ─────────────────────────────────────────────────────────────
# LOCATION ANALYSIS
# ─────────────────────────────────────────────────────────────

st.subheader("Customer Distribution")

tab1, tab2 = st.tabs(["By State", "By City"])

# STATE
with tab1:
    state_df = (
        fdf.groupby("state")
        .size()
        .nlargest(15)
        .sort_values()
        .reset_index(name="customers")
    )

    fig = px.bar(
        state_df,
        x="customers",
        y="state",
        orientation="h",
        title="Top States by Customers",
        color="customers",
        color_continuous_scale="Oranges"
    )

    st.plotly_chart(fig, use_container_width=True)

# CITY
with tab2:
    if city_col:
        city_df = (
            fdf.groupby(city_col)
            .size()
            .nlargest(20)
            .sort_values()
            .reset_index(name="customers")
        )

        fig = go.Figure(go.Bar(
            y=city_df[city_col],
            x=city_df["customers"],
            orientation="h",
            marker_color=AMBER
        ))

        fig.update_layout(**LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# PAYMENT ANALYSIS
# ─────────────────────────────────────────────────────────────

st.subheader("Average order value per payment method")

pay_df = None

if pay_col:
    pay_df = (
        fdf.groupby(pay_col)
        .agg(
            revenue=("totalamount", "sum"),
            orders=("totalamount", "count"),
            aov=("totalamount", "mean")
        )
        .reset_index()
        .sort_values("aov", ascending=False)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pay_df[pay_col],
        y=pay_df["aov"],
        name="AOV",
        marker_color=AMBER
    ))

    fig.add_trace(go.Scatter(
        x=pay_df[pay_col],
        y=pay_df["revenue"],
        name="Revenue",
        yaxis="y2",
        line=dict(color=DARK, dash="dot")
    ))

    fig.update_layout(
        **LAYOUT,
        yaxis=dict(title="AOV"),
        yaxis2=dict(overlaying="y", side="right"),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Payment column not found")
