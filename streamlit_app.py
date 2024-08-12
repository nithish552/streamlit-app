import streamlit as st
import pandas as pd
from session import get_session,set_session,clear_session
import pymongo
st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout='centered'
)

import base64


img_url = "https://drive.google.com/file/d/1bVsrLn8xozPadpPRL2P6EU2zv0a8S-ab/view?usp=sharing"
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{img_url}");
    background-size: cover;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
    <style>
    .header {
        font-size: 45px;
        font-weight: bold;
        color: #444444;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f0f0f0; /* Light ash-gray background */
        border-radius: 10px; /* Optional: Rounded corners */
    }
    .sub-header {
        font-size: 30px;
        color: #666666;
        text-align: center;
        margin-bottom: 40px;
    }
    .button {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .button > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
    }
    .button > button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the headers
st.markdown('<div class="header">Welcome To</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Patient Adherence Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Add the "Sign in" button in the first column
with col1:
    if st.button('Sign in'):
        st.switch_page('pages/login.py')

# Add the "Sign up" button in the second column
with col2:
    if st.button('Sign up'):
        st.switch_page('pages/signup.py')
def home_page():
    get_session()
    if st.session_state.get('logged_in'):
        st.write(f"Welcome, {st.session_state['username']}!")
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Main Page", "Logout"])
        
        if page == "Main Page":
            st.switch_page("pages/patient_details.py")  
        elif page == "Logout":
            clear_session()
            st.switch_page("pages/login.py") 
    else:
        st.sidebar.empty()

if __name__ == "__main__":
    home_page()
