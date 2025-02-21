import streamlit as st
import os
import pickle
from chatbot import chatbot
import subprocess

st.set_page_config(page_title="AI Chatbot", layout="wide")

st.title("AI Chatbot")
st.write("Ask me anything about AI!")

# Load and Save Chat History with pickle
def save_chat_history():
    with open("chat_history.pkl", "wb") as f:
        pickle.dump(st.session_state["messages"], f)

def load_chat_history():
    if os.path.exists("chat_history.pkl"):
        with open("chat_history.pkl", "rb") as f:
            return pickle.load(f)
    return []

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF to add to the knowledge base", type=["pdf"])
if uploaded_file:
    save_path = os.path.join("./data", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"{uploaded_file.name} uploaded! Processing...")

    with st.spinner("Updating knowledge base..."):
        subprocess.run(["python", "prepare_vectordb.py"], check=True)

    st.success("PDF added successfully! Restart the chatbot to query the new data.")

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Type your question here...")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            for chunk in chatbot.stream_response(query):
                full_response += chunk
                response_container.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    save_chat_history()
