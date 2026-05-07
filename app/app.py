import streamlit as st

# Define pages from local files or functions
page1 = st.Page("pages/overview.py", title="Executive OverView", icon="📊")
page2 = st.Page("pages/analysis.py", title="Sales Analysis", icon="📈")
page3 = st.Page("pages/customer_insights.py", title="Customer insights", icon="🕵🏻")
page4 = st.Page("pages/product_performance.py", title="Product Performance", icon="⚡")
page5 = st.Page("pages/forecasting.py", title="Revenue Forecasting", icon="📶")

# Initialize navigation
pg = st.navigation([page1, page2, page3, page4, page5])

# Run the selected page
pg.run()
