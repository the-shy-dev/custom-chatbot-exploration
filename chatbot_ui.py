import streamlit as st
from chatbot import chatbot

st.set_page_config(page_title="Custom chatbot", layout="wide")

st.title("Custom chatbot")
st.write("Ask me anything about the data you provided!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Type your question here...")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in chatbot.stream_response(query):
            full_response += chunk
            response_container.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
