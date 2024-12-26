# PDF Whisperer: Chat with Your Documents 📚

**PDF Whisperer** is a powerful and user-friendly chatbot application that allows you to interact with your PDF documents. Leveraging state-of-the-art LLMs and vector databases, PDF Whisperer can process uploaded PDFs, create embeddings, and answer user questions based on the document's content.

## Features

- **PDF Upload and Processing**: Upload one or more PDF files to extract and index their content.
- **Vector Search with FAISS**: Store and retrieve embeddings for efficient document-based Q&A.
- **AI-Powered Q&A**: Uses Google Generative AI (Gemini Pro) to answer questions based on the document context.
- **Interactive Frontend**: Built with Streamlit for a conversational and intuitive user experience.
- **Customizable**: Easily extendable to support more features and other document formats.

---

## Tech Stack

- **Backend**: FastAPI for building the RESTful API.
- **Frontend**: Streamlit for the interactive chat interface.
- **LLM Integration**: Google Generative AI for generating embeddings and responses.
- **Vector Database**: FAISS for efficient similarity search and storage.
- **PDF Parsing**: PyPDF2 for extracting text from PDF documents.

---

## Installation

### Prerequisites

1. Python 3.8+ installed on your system.
2. A valid Google API key for accessing Google Generative AI services.

### Steps

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/pdf-whisperer.git
    cd pdf-whisperer
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY=your_google_api_key
    ```
   
5. In a new terminal, start the Streamlit frontend:
    ```bash
    streamlit run main.py
    ```

---

## Usage

1. **Upload PDFs**: Use the sidebar in the Streamlit interface to upload your PDF files and process them.
2. **Ask Questions**: Enter your question in the chat input box to get answers based on the uploaded documents.
3. **Clear Index**: If needed, clear the FAISS index using the provided button in the sidebar.

---

## Folder Structure

```plaintext
pdf-whisperer/
├── app.py               # Streamlit frontend
├── backend.py           # FastAPI backend
├── main.py              # Main File. 
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── README.md            # Project documentation
├── faiss_index/         # FAISS index files
└── ...
