import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Helper to clear FAISS index
def clear_faiss_index(index_folder="faiss_index"):
    if os.path.exists(index_folder):
        index_file = os.path.join(index_folder, "index.faiss")
        metadata_file = os.path.join(index_folder, "index.pkl")

        files_deleted = False

        if os.path.exists(index_file):
            os.remove(index_file)
            files_deleted = True

        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            files_deleted = True

        if files_deleted:
            return True
        else:
            return False
    else:
        return False


# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


# Create or update FAISS index
def get_vector_store(text_chunks, clear_existing=False):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_folder = "faiss_index"

    if clear_existing:
        if clear_faiss_index(index_folder):
            st.info("Existing FAISS index cleared.")

    index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(metadata_file):
        vector_store = FAISS.load_local(
            index_folder, embeddings, allow_dangerous_deserialization=True
        )
        vector_store.add_texts(text_chunks)
    else:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local(index_folder)


# Load conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a helpful and detailed assistant. Your goal is to answer user questions based on the provided context as elaborately as possible. Follow these rules carefully:

    1. If the user asks a general question (e.g., "Can you explain more about the files?"), summarize the content of the uploaded files in detail based on the context available. Provide a thorough explanation of the text, key topics, and insights from the context.

    2. If the user asks a specific question:
       - If the answer exists in the provided context, answer as comprehensively and accurately as possible, including all relevant details. Be elaborate and provide examples or additional context if necessary.
       - If the answer is not found in the provided context, respond with: "The answer is not available in the provided context." Do not attempt to guess or fabricate information.

    3. Never include information outside the provided context.

    4. Be polite, professional, and clear in your responses.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Generate a response to the user's question
def get_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_folder = "faiss_index"
    index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
        st.warning("FAISS index not found.")
        return "Please upload and process files first."

    new_db = FAISS.load_local(
        index_folder, embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response["output_text"]


def main():
    st.set_page_config(page_title="PDF Whisperer", layout="wide")
    st.header("Chat with Your Documents 📚")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        clear_index = st.button("Clear FAISS Index")
        if clear_index:
            if clear_faiss_index():
                st.success("FAISS index cleared successfully!")
            else:
                st.info("No FAISS index found to clear.")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, clear_existing=False)
                st.success("Processing complete!")

    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_question)
                st.markdown(response)

        # Save assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
