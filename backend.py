import os
from typing import List
from PyPDF2 import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.responses import JSONResponse
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, UploadFile, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


# Helper Functions
def clear_faiss_index(index_folder="faiss_index"):
    if os.path.exists(index_folder):
        index_file = os.path.join(index_folder, "index.faiss")
        metadata_file = os.path.join(index_folder, "index.pkl")

        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        return True
    return False


def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, clear_existing=False):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_folder = "faiss_index"

    if clear_existing:
        clear_faiss_index(index_folder)

    index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(metadata_file):
        vector_store = FAISS.load_local(index_folder, embeddings)
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


# API Endpoints
@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile]):
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks, clear_existing=False)
        return JSONResponse(content={"message": "PDFs processed successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_index/")
async def clear_index():
    if clear_faiss_index():
        return {"message": "FAISS index cleared successfully!"}
    else:
        return {"message": "No FAISS index to clear."}


@app.post("/ask/")
async def ask_question(data: str):
    question = data
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_folder = "faiss_index"
    index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
        raise HTTPException(
            status_code=400,
            detail="FAISS index not found. Please upload and process files first.",
        )

    new_db = FAISS.load_local(index_folder, embeddings)
    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    print("Response:", response, "\n\n\n")
    return {"answer": response["output_text"]}