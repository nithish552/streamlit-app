import streamlit as st


def set_session(username):
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    
def clear_session():
    st.session_state['logged_in'] = False
    st.session_state.pop('username', None)
    

def get_session():
    if 'logged_in' in st.session_state and st.session_state['logged_in'] == True:
        st.session_state['logged_in'] = True
        st.session_state['username'] =st.session_state['username']
    else:
        st.session_state['logged_in'] = False
