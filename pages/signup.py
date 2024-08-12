import streamlit as st
from auth import signup_user
from session import set_session
from time import sleep

st.markdown(
        """
        <style>
        .streamlit-expander, .streamlit-container {
            width: 100%;
            max-width: 100%;
        }
        .stForm {
            width: 100%;
            max-width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def signup_page():
    st.title("Sign Up")
    with st.form(key='signup_form'):
        email=st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Sign Up")
        
        if submit_button:
            if signup_user(email,password,username):
                set_session(username)
                st.success("Account created successfully!")
                sleep(1)
                st.switch_page("pages/patient_details.py")  # Switch to the login page
            else:
                st.error("Email already exists.")

if __name__ == "__main__":
    signup_page()
