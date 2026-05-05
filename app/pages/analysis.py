import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.title("Sales Analysis")
st.caption("Revenue breakdown, discount impact, shipping, and seasonal patterns")
st.divider()

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] { border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricLabel"] { font-size: 12px; color: #888; }
[data-testid="stMetricValue"] { font-size: 28px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

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


# ── Tabs: Revenue Breakdown ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["By Category", "By Brand", "By Payment Method"])

with tab1:
    cat_rev = (
        fdf.groupby("Category")["TotalAmount"]
        .sum()
        .reset_index()
        .sort_values("TotalAmount", ascending=False)
    )
    fig1 = px.bar(
        cat_rev, x="Category", y="TotalAmount",
        title="Revenue by Category",
        color="TotalAmount",
        color_continuous_scale=["#FFE8C2", "#FF9900"],
        template="plotly_white"
    )
    fig1.update_layout(margin=dict(l=0, r=0, t=36, b=0),
                       height=300, coloraxis_showscale=False,
                       xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    brand_rev = (
        fdf.groupby("Brand")["TotalAmount"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    fig2 = px.bar(
        brand_rev, x="TotalAmount", y="Brand",
        orientation="h",
        title="Top 10 Brands by Revenue",
        color="TotalAmount",
        color_continuous_scale=["#FFE8C2", "#FF9900"],
        template="plotly_white"
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=36, b=0),
                       height=300, coloraxis_showscale=False,
                       xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    pay_rev = (
        fdf.groupby("PaymentMethod")["TotalAmount"]
        .sum()
        .reset_index()
    )
    fig3 = px.pie(
        pay_rev,
        values="TotalAmount",
        names="PaymentMethod",
        title="Revenue by Payment Method",
        hole=0.55,
        color_discrete_sequence=["#FF9900","#FFB347","#FFC875","#FFD9A0","#FFE8C2"]
    )
    fig3.update_traces(textinfo="percent",
                       hovertemplate="%{label}: $%{value:,.0f}")
    fig3.update_layout(margin=dict(l=0, r=0, t=36, b=0),
                       height=300)
    st.plotly_chart(fig3, use_container_width=True)

st.write("")

# ── Row 2: Discount Impact + Shipping Cost ────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    fig4 = px.scatter(
        fdf,
        x="Discount",
        y="TotalAmount",
        title="Discount vs Revenue",
        opacity=0.6,
        template="plotly_white"
    )
    fig4.update_layout(
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        xaxis_title=None,
        yaxis_title=None
    )
    st.plotly_chart(fig4, use_container_width=True)

with c2:
    fig5 = px.box(
        fdf,
        x="Country",
        y="ShippingCost",
        title="Shipping Cost Distribution by Country",
        template="plotly_white"
    )
    fig5.update_layout(
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        xaxis_title=None,
        yaxis_title=None
    )
    st.plotly_chart(fig5, use_container_width=True)

st.write("")

# ── Row 3: Monthly Heatmap ────────────────────────────────────────────────
heat_df = fdf.copy()
heat_df["Month"] = heat_df["OrderDate"].dt.month_name()

pivot = (
    heat_df.pivot_table(
        index="Month",
        columns="Category",
        values="TotalAmount",
        aggfunc="sum"
    )
)

fig6 = px.imshow(
    pivot,
    aspect="auto",
    title="Monthly Revenue Heatmap (Category vs Month)",
    color_continuous_scale=["#FFE8C2", "#FF9900"]
)

fig6.update_layout(
    margin=dict(l=0, r=0, t=36, b=0),
    height=500,  # ↑ increase height (main factor)
    xaxis_title=None,
    yaxis_title=None
)

# Optional: make labels clearer (helps perception of size)
fig6.update_xaxes(tickangle=45)

st.plotly_chart(fig6, use_container_width=True)