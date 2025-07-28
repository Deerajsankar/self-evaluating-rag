# 🤖 AI Document Analysis Tool with Confidence Score

This project provides an interactive web application for analyzing documents using a Retrieval-Augmented Generation (RAG) pipeline. The system features a self-evaluation mechanism where the AI provides a **confidence score** for each answer, indicating how well its response is supported by the provided source documents.

The application is built with Python, LangChain, Google Gemini, and Gradio.


## Features
-   **Document Upload:** Supports both PDF (`.pdf`) and plain text (`.txt`) files.
-   **RAG Pipeline:** Uses a sophisticated pipeline to retrieve relevant context and generate answers.
-   **Self-Evaluation:** Each answer is automatically critiqued by the AI to produce a confidence score, helping to identify potential hallucinations.
-   **Efficient Caching:** A two-step process ensures that the time-consuming document embedding happens only once, making subsequent queries fast.
-   **Interactive UI:** A clean and professional user interface built with Gradio.

## Tech Stack
-   **Backend:** Python
-   **LLM Framework:** LangChain
-   **Language Model:** Google Gemini (`gemini-1.5-flash`)
-   **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
-   **Vector Store:** FAISS (Facebook AI Similarity Search)
-   **Web UI:** Gradio

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/document-analysis-app.git](https://github.com/your-username/document-analysis-app.git)
cd document-analysis-app
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# For MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
You need a Google AI API key to use the Gemini model.

a. Create a file named `.env` in the root directory of the project.
b. Add your API key to this file as follows:
```
GOOGLE_API_KEY="your_api_key_here"
```

## How to Run
Once the setup is complete, you can launch the Gradio application with the following command:
```bash
gradio app.py
```
This will start a local web server. Open the URL provided in your terminal (usually `http://127.0.0.1:7860`) in your web browser to use the application.

## How to Use
The application has a simple two-step workflow:
1.  **Build Knowledge Base:** Upload one or more documents and click the "Build Knowledge Base" button. This will be slow the first time as the documents are processed and indexed.
2.  **Ask a Question:** Once the knowledge base is ready, type your question into the text box and click the "Get Answer" button. This step is fast and can be repeated multiple times.
