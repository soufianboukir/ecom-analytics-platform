# E-Commerce Analytics & Forecasting Platform

An end-to-end data science and business intelligence system built on 100,000 Amazon-style sales transactions — combining interactive dashboards, customer analytics, product performance analysis, and multi-model revenue forecasting.

## 📄 Report
- Full analysis report: [PDF](https://github.com/soufianboukir/ecom-analytics-platform/blob/main/reports/main.pdf)
- Streamlit dashboard: [Live App](https://ecom-analytics-forecasting-platform.streamlit.app/)

<img width="2048" height="1032" alt="image" src="https://github.com/user-attachments/assets/a8200779-4919-4896-9408-4b345627051e" />


## Overview

This project develops a **5-page interactive Streamlit dashboard** that transforms raw e-commerce transactional data into actionable business intelligence. It covers the full data science pipeline:

- **Exploratory Data Analysis** — distributions, correlations, trends
- **Customer Analytics** — lifetime value, geographic distribution, payment behavior
- **Product Performance** — revenue, margins, brand comparison, drilldown
- **Revenue Forecasting** — 4 models benchmarked, XGBoost selected for 12-month forecast
- **Return Analysis** — country × category heatmap, top returned products

The system is designed to answer real business questions:
- Which customers, products, and geographies drive the most revenue?
- Do discounts actually increase revenue?
- What will revenue look like over the next 12 months?
- Which products are being returned most and why?


---

## Project Structure

```
ecom-analytics-platform/
│
├── app/
│   ├── app.py                        # Main Streamlit entry point
│   └── pages/
│       ├── overview.py   # Page 1 — KPIs, revenue trend, top countries
│       ├── analysis.py       # Page 2 — Revenue by category, discounts, shipping
│       ├── customer_insights.py    # Page 3 — LTV, geography, payment methods
│       ├── product_performance.py  # Page 4 — Products, margins, brands, drilldown
│       └── forecasting.py          # Page 5 — Multi-model forecasting system
│
├── data/
│   ├── raw/
│   │   └── amazon_sales.csv          # Original dataset
│   └── processed/
│       ├── amazon_sales_final.csv    # Cleaned & feature-engineered dataset
│       └── amazon_sales.py
│
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb 
│   └── 03_feature_engineering.ipynb  
│
├── requirements.txt                  # Python dependencies
├── README.md                 
└── report/
    └── main.pdf                      # Full academic report
```

---

## Dashboard Pages

### Page 1 — Executive Overview
High-level business snapshot with 4 KPI cards, monthly revenue trend, top 5 countries by revenue, top 5 categories donut chart, and order status breakdown.

| KPI | Value |
|---|---|
| Total Revenue | $91,825,648 |
| Total Orders | 100,000 |
| Avg Order Value | $918.26 |
| Return Rate | 6.2% |

---

### Page 2 — Sales Analysis
- Revenue breakdown by Category, Brand, and Payment Method
- Discount vs Revenue scatter analysis
- Shipping cost distribution by country (box plots)
- Seasonal revenue patterns

---

### Page 3 — Customer Insights
- Top 20 customers by Lifetime Value (LTV) — horizontal gradient bar chart
- Customer geographic distribution — US choropleth map + city bar chart
- Average order value by payment method — grouped bar + revenue share donut
- Export buttons for LTV and payment summary CSVs

---

### Page 4 — Product Performance
- Best-selling products — Top 20 by revenue and by quantity (tabbed)
- Category margin analysis — Revenue vs Shipping vs Tax vs Margin grouped bar
- Brand bubble chart — Avg unit price vs Avg quantity (bubble size = revenue)
- Product drilldown — search bar → KPIs + monthly sparkline + raw orders table
- Return analysis — By country, by category, country × category heatmap, top 10 returned products

---

### Page 5 — Revenue Forecasting System
- 4 models: Naive baseline, Linear Regression, XGBoost, Prophet
- Time-based train/test split — last 6 months as holdout
- Actual vs Predicted chart (per model selector)
- All models comparison chart
- Model performance table — MAE, RMSE, R², MAPE
- XGBoost 12-month recursive future forecast with downloadable CSV

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/soufianboukir/ecom-analytics-platform.git
cd ecom-analytics-platform
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

### Sidebar Filters
- **Date Range** — Filter all pages by order date
- **Category** — Filter by product category (available on relevant pages)

built with ❤️ by **soufian**.
