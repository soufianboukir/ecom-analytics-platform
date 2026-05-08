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

PALETTE = ["#0b69f5","#3b82f6","#10b981","#8b5cf6","#ef4444","#f97316","#14b8a6","#ec4899"]
LAYOUT  = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", color="#1f2937"),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=45, b=10),
)

def hbar(df_plot, x, y, title, color="#0bdef5", height=420):
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
            hbar(top_rev, "Revenue", PRODUCT_COL, f"Top {top_n} Products by Revenue", color="#3b82f6"),
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
        xaxis_title="Category", yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(150,150,150,0.2)"),
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
        color_continuous_scale=[[0,"#8ad1fd"],[0.5,"#0b7cf5"],[1,"#0e3892"]],
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
                <p style="margin:4px 0 0;font-size:24px;font-weight:700;">{val}</p>
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
            line=dict(color="#0b49f5", width=2),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(116,158,245,0.08)",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
        fig_drill.update_layout(
            **LAYOUT, height=300,
            title=f"Monthly Revenue — {selected}",
            xaxis_title="Month",
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(150,150,150,0.2)"),
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






import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "amazon_sales_final.csv"

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        dayfirst=True,
        errors="coerce"
    )

    return df.dropna(subset=["OrderDate"])


df = load_data()

# ─────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────
def find_col(candidates):
    return next(
        (c for c in df.columns if c.lower() in candidates),
        None
    )

STATUS_COL = find_col(["orderstatus", "order_status", "status"])
COUNTRY_COL = find_col(["country"])
CATEGORY_COL = find_col(["category"])
AMOUNT_COL = find_col(["totalamount", "total_amount", "amount"])
PRODUCT_COL = find_col([
    "productname"
])
# ─────────────────────────────────────────────────────────────
# PAGE TITLE
# ─────────────────────────────────────────────────────────────
st.subheader("Return Analysis — By Country & Category")

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
if not STATUS_COL:
    st.error("No order status column found.")
    st.stop()

if not AMOUNT_COL:
    st.error("No revenue column found.")
    st.stop()

# safer string handling
returned = df[
    df[STATUS_COL]
    .astype(str)
    .str.lower()
    .str.contains("return", na=False)
]

if returned.empty:
    st.warning("No returned orders found in the dataset.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────
total_orders = len(df)
returned_orders = len(returned)

return_rate = (
    (returned_orders / total_orders) * 100
    if total_orders > 0 else 0
)

lost_revenue = returned[AMOUNT_COL].fillna(0).sum()

k1, k2, k3 = st.columns(3)

kpi_data = [
    (k1, "Total Returned Orders", f"{returned_orders:,}", "#ef4444"),
    (k2, "Return Rate", f"{return_rate:.2f}%", "#f97316"),
    (k3, "Est. Lost Revenue", f"${lost_revenue:,.0f}", "#8b5cf6"),
]

for box, label, value, color in kpi_data:
    box.markdown(
        f"""<div style="border:1px solid #e5e7eb;border-top:3px solid {color};border-radius:8px;padding:14px 18px;"><p style="margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:600;">{label}</p><p style="margin:4px 0 0;font-size:24px;font-weight:700;">{value}</p></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COMMON LAYOUT
# ─────────────────────────────────────────────────────────────
LAYOUT = dict(
    template="plotly_white",
    font=dict(
        family="Inter, sans-serif",
        color="#1f2937"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=45, b=10),
)

tab1, tab2, tab3, tab4 = st.tabs([
    "By Country",
    "By Category",
    "Country x Category Heatmap",
    "Top Returned Products"
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — COUNTRY
# ─────────────────────────────────────────────────────────────
with tab1:

    if not COUNTRY_COL:
        st.info("No country column found.")

    else:

        country_ret = (
            returned.groupby(COUNTRY_COL)
            .agg(
                Returns=(AMOUNT_COL, "count"),
                LostRevenue=(AMOUNT_COL, "sum")
            )
            .join(
                df.groupby(COUNTRY_COL)
                .size()
                .rename("Total")
            )
            .reset_index()
        )

        country_ret["ReturnRate%"] = (
            country_ret["Returns"] /
            country_ret["Total"]
        ).fillna(0) * 100

        country_ret["ReturnRate%"] = (
            country_ret["ReturnRate%"]
            .round(2)
        )

        country_ret = country_ret.sort_values(
            "Returns",
            ascending=True
        )

        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            y=country_ret[COUNTRY_COL],
            x=country_ret["Returns"],
            orientation="h",

            marker=dict(
                color=country_ret["Returns"],
                colorscale=[
                    [0, "#fecaca"],
                    [1, "#b91c1c"]
                ],
                showscale=False
            ),

            customdata=country_ret[
                ["LostRevenue", "ReturnRate%"]
            ].values,

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Returns: %{x:,}<br>"
                "Lost Revenue: $%{customdata[0]:,.0f}<br>"
                "Return Rate: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
        ))

        fig1.update_layout(
            **LAYOUT,
            height=320,
            title="Returned Orders by Country",
            xaxis=dict(
                title="Returns",
                gridcolor="#f3f4f6"
            ),
            yaxis=dict(title="")
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
# TAB 2 — CATEGORY
# ─────────────────────────────────────────────────────────────
with tab2:

    if not CATEGORY_COL:
        st.info("No category column found.")

    else:

        cat_ret = (
            returned.groupby(CATEGORY_COL)
            .agg(
                Returns=(AMOUNT_COL, "count"),
                LostRevenue=(AMOUNT_COL, "sum")
            )
            .join(
                df.groupby(CATEGORY_COL)
                .size()
                .rename("Total")
            )
            .reset_index()
        )

        cat_ret["ReturnRate%"] = (
            cat_ret["Returns"] /
            cat_ret["Total"]
        ).fillna(0) * 100

        cat_ret["ReturnRate%"] = (
            cat_ret["ReturnRate%"]
            .round(2)
        )

        cat_ret = cat_ret.sort_values(
            "Returns",
            ascending=False
        )

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=cat_ret[CATEGORY_COL],
            y=cat_ret["Returns"],
            name="Returns",
            marker_color="#ef4444"
        ))

        fig2.add_trace(go.Scatter(
            x=cat_ret[CATEGORY_COL],
            y=cat_ret["ReturnRate%"],
            name="Return Rate %",
            mode="lines+markers",
            yaxis="y2",
            line=dict(
                color="#1f2937",
                width=2,
                dash="dot"
            )
        ))

        fig2.update_layout(
            **LAYOUT,
            height=360,
            title="Returns & Return Rate by Category",

            xaxis=dict(
                title="Category",
                tickangle=-15
            ),

            yaxis=dict(
                title="Returns",
                gridcolor="#f3f4f6"
            ),

            yaxis2=dict(
                title="Return Rate (%)",
                overlaying="y",
                side="right",
                ticksuffix="%"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
# TAB 3 — HEATMAP
# ─────────────────────────────────────────────────────────────
with tab3:

    if not COUNTRY_COL or not CATEGORY_COL:
        st.info("Country or Category column not found.")

    else:

        pivot = (
            returned.groupby(
                [COUNTRY_COL, CATEGORY_COL]
            )
            .size()
            .reset_index(name="Returns")
            .pivot(
                index=COUNTRY_COL,
                columns=CATEGORY_COL,
                values="Returns"
            )
            .fillna(0)
            .astype(int)
        )

        fig3 = go.Figure()

        fig3.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,

            colorscale=[
                [0, "#fff7ed"],
                [0.5, "#f97316"],
                [1, "#7c2d12"]
            ],

            hovertemplate=(
                "<b>%{y} — %{x}</b><br>"
                "Returns: %{z:,}"
                "<extra></extra>"
            ),

            colorbar=dict(
                title="Returns",
                thickness=14,
                len=0.7
            )
        ))

        fig3.update_layout(
            **LAYOUT,
            height=380,
            title="Return Count Heatmap — Country × Category",

            xaxis=dict(
                title="Category",
                tickangle=-15
            ),

            yaxis=dict(title="Country")
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.caption(
            "Darker cells indicate higher return concentration."
        )



# ─────────────────────────────────────────────────────────────
# TAB 4 — TOP RETURNED PRODUCTS
# ─────────────────────────────────────────────────────────────
with tab4:

    if not PRODUCT_COL:
        st.info("No product column found.")

    else:

        top_products = (
            returned.groupby(PRODUCT_COL)
            .agg(
                Returns=(AMOUNT_COL, "count"),
                LostRevenue=(AMOUNT_COL, "sum")
            )
            .reset_index()
            .sort_values("Returns", ascending=False)
            .head(10)
        )

        fig4 = go.Figure()

        fig4.add_trace(go.Bar(
            y=top_products[PRODUCT_COL],
            x=top_products["Returns"],
            orientation="h",

            marker=dict(
                color=top_products["Returns"],
                colorscale=[
                    [0, "#fde68a"],
                    [1, "#dc2626"]
                ],
                showscale=False
            ),

            customdata=top_products["LostRevenue"],

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Returns: %{x:,}<br>"
                "Lost Revenue: $%{customdata:,.0f}"
                "<extra></extra>"
            )
        ))

        fig4.update_layout(
            **LAYOUT,

            height=450,

            title="Top 10 Most Returned Products",

            xaxis=dict(
                title="Number of Returns",
                gridcolor="#f3f4f6"
            ),

            yaxis=dict(
                title="",
                autorange="reversed"
            )
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.dataframe(
            top_products.rename(columns={
                PRODUCT_COL: "Product",
                "LostRevenue": "Lost Revenue ($)"
            })
            .style.format({
                "Returns": "{:,}",
                "Lost Revenue ($)": "${:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Products with the highest number of returns and estimated lost revenue."
        )