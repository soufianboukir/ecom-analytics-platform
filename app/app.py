import streamlit as st

# Define pages from local files or functions
page1 = st.Page("pages/overview.py", title="Executive OverView", icon="📊")
page2 = st.Page("pages/analysis.py", title="Sales Analysis", icon="📈")
page3 = st.Page("pages/settings.py", title="Settings", icon="⚙️")

# Initialize navigation
pg = st.navigation([page1, page2, page3])

# Run the selected page
pg.run()
