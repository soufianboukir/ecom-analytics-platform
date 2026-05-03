import streamlit as st

# This code runs whenever the app starts or a user interacts with a widget
st.title("Hello Streamlit!")
st.write("This is my basic entry point script.")

name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")