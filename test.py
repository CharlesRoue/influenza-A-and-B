import streamlit as st

button_clicked = st.button("Click Me")

if button_clicked:
    st.write("Button was clicked!")