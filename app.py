import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="PDF Whisperer", layout="wide")
st.header("Chat with Your Documents 📚")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on the Submit & Process Button",
        accept_multiple_files=True,
    )

    if st.button("Clear FAISS Index"):
        response = requests.post(f"{API_URL}/clear_index/")
        if response.status_code == 200:
            st.success(response.json().get("message", "Index cleared."))
        else:
            st.error("Failed to clear FAISS index.")

    if st.button("Submit & Process"):
        if pdf_docs:
            files = [
                ("files", (pdf.name, pdf.getvalue(), pdf.type)) for pdf in pdf_docs
            ]
            response = requests.post(f"{API_URL}/upload/", files=files)
            if response.status_code == 200:
                st.success(response.json().get("message", "Processing complete!"))
            else:
                st.error("Failed to process PDFs.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask a question")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    assistant_response = "Sorry, I couldn't get a response. Please try again later."
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_question = str(user_question)
            print(f"Received questiona: {user_question}")
            response = requests.post(
                f"{API_URL}/ask/", data=user_question
            )
            print(response)
            if response.status_code == 200:
                assistant_response = response.json().get("answer", "No response.")
                st.markdown(assistant_response)
            else:
                st.error("Failed to get a response. Please try again.")

        # Save assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
