import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.title("Product Performance")
st.caption("Revenue · Margins · Brand comparison · Product drilldown")
st.divider()

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")
    df["Margin"]    = df["TotalAmount"] - df.get("ShippingCost", 0) - df.get("Tax", 0)
    return df.dropna(subset=["OrderDate"])

df = load()

# ── column resolver ──────────────────────────────────────────
def col(candidates: list[str]) -> str | None:
    return next((c for c in df.columns if c.lower() in candidates), None)

PRODUCT_COL  = col(["productname", "product_name", "product"])
CATEGORY_COL = col(["category"])
BRAND_COL    = col(["brand"])
QTY_COL      = col(["quantity", "qty"])
PRICE_COL    = col(["unitprice", "unit_price", "price"])
SHIP_COL     = col(["shippingcost", "shipping_cost", "shipping"])
TAX_COL      = col(["tax"])

# ── sidebar filter ───────────────────────────────────────────
st.sidebar.header("Filters")
date_input = st.sidebar.date_input(
    "Date Range",
    value=[df["OrderDate"].min().date(), df["OrderDate"].max().date()],
)
start_date, end_date = (date_input if len(date_input) == 2 else (date_input, date_input))

if CATEGORY_COL:
    cats = st.sidebar.multiselect("Category", sorted(df[CATEGORY_COL].dropna().unique()), default=[])

fdf = df[df["OrderDate"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
if CATEGORY_COL and cats:
    fdf = fdf[fdf[CATEGORY_COL].isin(cats)]

if fdf.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SHARED STYLES
# ─────────────────────────────────────────────────────────────

PALETTE = ["#f59e0b","#3b82f6","#10b981","#8b5cf6","#ef4444","#f97316","#14b8a6","#ec4899"]
LAYOUT  = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", color="#1f2937"),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=45, b=10),
)

def hbar(df_plot, x, y, title, color="#f59e0b", height=420):
    fig = go.Figure(go.Bar(
        x=df_plot[x], y=df_plot[y], orientation="h",
        marker=dict(
            color=[f"rgba(245,158,11,{0.4 + 0.6*i/max(len(df_plot)-1,1)})" for i in range(len(df_plot))],
            line=dict(width=0),
        ),
        hovertemplate=f"<b>%{{y}}</b><br>{x}: %{{x:,.2f}}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT, height=height, title=title,
                      xaxis=dict(gridcolor="#f3f4f6"), yaxis=dict(title=""))
    return fig

# ─────────────────────────────────────────────────────────────
# 1. BEST-SELLING PRODUCTS
# ─────────────────────────────────────────────────────────────

st.subheader("Best-Selling Products — Top 20")

if not PRODUCT_COL:
    st.info("No product column found.")
else:
    top_n = 20
    agg   = fdf.groupby(PRODUCT_COL).agg(
        Revenue  =("TotalAmount", "sum"),
        Quantity =(QTY_COL,       "sum") if QTY_COL else ("TotalAmount", "count"),
        Orders   =("TotalAmount", "count"),
    ).reset_index()

    tab_rev, tab_qty = st.tabs(["By Revenue", "By Quantity"])

    with tab_rev:
        top_rev = agg.nlargest(top_n, "Revenue").sort_values("Revenue")
        st.plotly_chart(
            hbar(top_rev, "Revenue", PRODUCT_COL, f"Top {top_n} Products by Revenue"),
            use_container_width=True,
        )

    with tab_qty:
        top_qty = agg.nlargest(top_n, "Quantity").sort_values("Quantity")
        st.plotly_chart(
            hbar(top_qty, "Quantity", PRODUCT_COL, f"Top {top_n} Products by Quantity Sold", color="#3b82f6"),
            use_container_width=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────
# 2. CATEGORY MARGIN ANALYSIS
# ─────────────────────────────────────────────────────────────

st.subheader("Category Margin Analysis")

if not CATEGORY_COL:
    st.info("No category column found.")
else:
    cat_agg = fdf.groupby(CATEGORY_COL).agg(
        Revenue     =("TotalAmount",  "sum"),
        Shipping    =(SHIP_COL,       "sum") if SHIP_COL else ("TotalAmount", lambda x: 0),
        Tax         =(TAX_COL,        "sum") if TAX_COL  else ("TotalAmount", lambda x: 0),
        Margin      =("Margin",       "sum"),
        Orders      =("TotalAmount",  "count"),
    ).reset_index()
    cat_agg["MarginPct"] = (cat_agg["Margin"] / cat_agg["Revenue"] * 100).round(2)
    cat_agg = cat_agg.sort_values("Margin", ascending=False)

    fig_cat = go.Figure()
    for trace, col_name, colour in [
        ("Revenue",  "Revenue",  "#3b82f6"),
        ("Shipping", "Shipping", "#ef4444"),
        ("Tax",      "Tax",      "#f97316"),
        ("Margin",   "Margin",   "#10b981"),
    ]:
        fig_cat.add_trace(go.Bar(
            name=trace, x=cat_agg[CATEGORY_COL], y=cat_agg[col_name],
            marker_color=colour,
            hovertemplate=f"{trace}: $%{{y:,.0f}}<extra></extra>",
        ))

    fig_cat.update_layout(
        **LAYOUT, barmode="group", height=380,
        title="Revenue vs Shipping vs Tax vs Margin by Category",
        xaxis_title="Category", yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Margin % summary table
    st.dataframe(
        cat_agg[[CATEGORY_COL, "Revenue", "Margin", "MarginPct", "Orders"]]
        .rename(columns={CATEGORY_COL: "Category", "MarginPct": "Margin %"})
        .style
        .format({"Revenue": "${:,.0f}", "Margin": "${:,.0f}", "Margin %": "{:.1f}%", "Orders": "{:,}"})
        .bar(subset=["Margin %"], color="#10b98155"),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# 3. BRAND COMPARISON
# ─────────────────────────────────────────────────────────────

st.subheader("Brand Comparison — Avg Unit Price vs Avg Quantity")

if not BRAND_COL:
    st.info("No brand column found.")
elif not PRICE_COL or not QTY_COL:
    st.info("UnitPrice or Quantity column not found.")
else:
    brand_agg = fdf.groupby(BRAND_COL).agg(
        AvgPrice  =(PRICE_COL, "mean"),
        AvgQty    =(QTY_COL,   "mean"),
        Revenue   =("TotalAmount", "sum"),
        Orders    =("TotalAmount", "count"),
    ).reset_index()

    top_brands = st.slider("Show top N brands by revenue", 5, 40, 20, step=5)
    brand_agg  = brand_agg.nlargest(top_brands, "Revenue")

    fig_brand = px.scatter(
        brand_agg,
        x="AvgPrice", y="AvgQty",
        size="Revenue", color="Revenue",
        hover_name=BRAND_COL,
        color_continuous_scale=[[0,"#fde68a"],[0.5,"#f59e0b"],[1,"#92400e"]],
        size_max=50,
        labels={"AvgPrice": "Avg Unit Price ($)", "AvgQty": "Avg Quantity Ordered"},
        hover_data={"Orders": True, "Revenue": ":$,.0f"},
    )
    fig_brand.update_layout(
        **LAYOUT, height=440,
        title=f"Brand Bubble Chart — Top {top_brands} by Revenue (bubble size = revenue)",
        coloraxis_colorbar=dict(title="Revenue", tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(fig_brand, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 4. PRODUCT DRILLDOWN
# ─────────────────────────────────────────────────────────────

st.subheader("Product Drilldown")

if not PRODUCT_COL:
    st.info("No product column found.")
else:
    search = st.text_input("Search product", placeholder="Type a product name…")
    all_products = sorted(fdf[PRODUCT_COL].dropna().unique())
    matches      = [p for p in all_products if search.lower() in p.lower()] if search else all_products
    selected     = st.selectbox("Select product", matches if matches else ["No match found"])

    if selected and selected != "No match found":
        pdf = fdf[fdf[PRODUCT_COL] == selected].copy()

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        metrics = [
            ("Total Revenue",   f"${pdf['TotalAmount'].sum():,.0f}",     "#f59e0b"),
            ("Total Orders",    f"{len(pdf):,}",                          "#3b82f6"),
            ("Avg Order Value", f"${pdf['TotalAmount'].mean():,.2f}",     "#10b981"),
            ("Avg Margin",      f"${pdf['Margin'].mean():,.2f}",          "#8b5cf6"),
        ]
        for col_st, (label, val, clr) in zip([k1, k2, k3, k4], metrics):
            col_st.markdown(
                f"""<div style="border:1px solid #e5e7eb;border-top:3px solid {clr};
                border-radius:8px;padding:14px 18px">
                <p style="margin:0;font-size:11px;color:#6b7280;text-transform:uppercase;
                letter-spacing:.06em;font-weight:600">{label}</p>
                <p style="margin:4px 0 0;font-size:24px;font-weight:700;color:#1f2937">{val}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Monthly revenue trend for selected product
        monthly = (
            pdf.set_index("OrderDate")["TotalAmount"]
            .resample("MS").sum()
            .reset_index()
            .rename(columns={"OrderDate": "Month", "TotalAmount": "Revenue"})
        )

        fig_drill = go.Figure()
        fig_drill.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
        fig_drill.update_layout(
            **LAYOUT, height=300,
            title=f"Monthly Revenue — {selected}",
            xaxis_title="Month",
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig_drill, use_container_width=True)

        # Raw orders table
        with st.expander("View individual orders"):
            display_cols = [c for c in ["OrderDate", CATEGORY_COL, BRAND_COL, QTY_COL,
                                         PRICE_COL, "TotalAmount", "Margin"] if c]
            st.dataframe(
                pdf[display_cols].sort_values("OrderDate", ascending=False)
                .style.format({
                    "TotalAmount": "${:,.2f}",
                    "Margin":      "${:,.2f}",
                    PRICE_COL:     "${:,.2f}" if PRICE_COL else "{}",
                }),
                use_container_width=True, hide_index=True,
            )