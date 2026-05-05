import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
st.set_page_config(page_title="Executive Overview", layout="wide", page_icon="📦")

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
    df = df.dropna(subset=["OrderDate"])  # drop rows where date failed to parse
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

st.title("Executive Overview")
st.caption(f"Amazon Sales · {len(fdf):,} orders · {date_range[0]} - {date_range[1]}")
st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue",    f"${fdf['TotalAmount'].sum():,.0f}")
k2.metric("Total Orders",     f"{len(fdf):,}")
k3.metric("Avg Order Value",  f"${fdf['TotalAmount'].mean():.2f}")
k4.metric("Return Rate",      "6.2%")

st.write("")

# ── Row 2: Trend + Top Countries ────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    monthly = (
        fdf.set_index("OrderDate")
           .resample("ME")["TotalAmount"]
           .sum()
           .reset_index()
    )
    monthly.columns = ["Month", "Revenue"]
    fig = px.area(
        monthly, x="Month", y="Revenue",
        title="Monthly Revenue Trend",
        color_discrete_sequence=["#FF9900"],
        template="plotly_white"
    )
    fig.update_traces(line_width=2, fillcolor="rgba(255,153,0,0.15)")
    fig.update_layout(margin=dict(l=0, r=0, t=36, b=0), height=220,
                      showlegend=False, yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    top_countries = (
        fdf.groupby("Country")["TotalAmount"]
           .sum().nlargest(5).reset_index()
           .sort_values("TotalAmount")
    )
    fig2 = px.bar(
        top_countries, x="TotalAmount", y="Country",
        orientation="h", title="Top 5 Countries by Revenue",
        color="TotalAmount", color_continuous_scale=["#FFE8C2", "#FF9900"],
        template="plotly_white"
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=36, b=0), height=220,
                       showlegend=False, coloraxis_showscale=False,
                       yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    top_cats = fdf.groupby("Category")["TotalAmount"].sum().nlargest(5).reset_index()
    fig3 = px.pie(
        top_cats, values="TotalAmount", names="Category",
        title="Top 5 Categories",
        hole=0.55,
        color_discrete_sequence=["#FF9900","#FFB347","#FFC875","#FFD9A0","#FFE8C2"],
    )
    fig3.update_traces(textinfo="percent", hovertemplate="%{label}: $%{value:,.0f}")
    fig3.update_layout(margin=dict(l=0, r=0, t=36, b=0), height=220,
                       legend=dict(font_size=11))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    status_counts = fdf["OrderStatus"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig4 = px.bar(
        status_counts, x="Count", y="Status",
        orientation="h", title="Order Status Breakdown",
        color="Status",
        color_discrete_map={
            "Delivered":"#4CAF50","Shipped":"#FF9900",
            "Processing":"#FFC107","Cancelled":"#F44336"
        },
        template="plotly_white"
    )
    fig4.update_layout(margin=dict(l=0, r=0, t=36, b=0), height=220,
                       showlegend=False, yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig4, use_container_width=True)