import streamlit as st
from auth import login_user
from session import clear_session
import base64

# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Path to your local image file
# img_path = 'imagefiles\local-doctor.jpg'  # Ensure this is the correct path to your image
# img_base64 = get_base64(img_path)

# # Custom CSS to add the background image
# page_bg_img = f'''
# <style>
# .stApp{{
#     background-image: url("data:image/jpg;base64,{img_base64}");
#     background-size: cover;
#     background-repeat: no-repeat;
# }}
# </style>
# '''

# # Inject the CSS with the background image into the Streamlit app
# st.markdown(page_bg_img, unsafe_allow_html=True)


def login_page():
    st.title("Login")
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.switch_page("pages/patient_details.py")
    with st.form(key='login_form'):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button and login_user(email,password):
            st.success("Login successful!")
            st.switch_page("pages/patient_details.py")  # Switch to the home page
        else:
            if email and password:
                st.error("Invalid username or password")
           

if __name__ == "__main__":
    login_page()
