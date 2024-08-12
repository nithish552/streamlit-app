import streamlit as st
import random
import time
import base64
from gradio_client import Client
@st.cache_resource
def getClient():
    return Client("yuva2110/vanilla-charbot")
client=getClient()
def getChat(question):
    result = client.predict(
            message=question,
            system_message="You are a friendly Chatbot.",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            api_name="/chat"
    )
    return result


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def add_bg_from_localchat(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .ScrollToBottomContainer{{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function to add background image to the sidebar
    def add_bg_to_sidebarchat(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Paths to your local images
        main_bg_pathchat = 'imagefiles\pexels-padrinan-255379.jpg'
        sidebar_bg_pathchat = 'imagefiles\pill-tablet-pharmacy-medicine.jpg'
    # Add background images
        add_bg_from_localchat(main_bg_pathchat)
        add_bg_to_sidebarchat(sidebar_bg_pathchat)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(getChat(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})