from pymongo import MongoClient
import streamlit as st


@st.cache_resource
def init_connection():
    return MongoClient(st.secrets["mongo"]['connection_string'])

client=init_connection()

@st.cache_resource
def getUser():
    users_collection=client.healthRecommend.users
    return users_collection
users_collection=getUser()
print(users_collection)