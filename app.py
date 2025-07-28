# Import all necessary libraries
import gradio as gr
import os
import tempfile
import shutil
from google.colab import userdata
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, conint
from langchain_core.output_parsers import JsonOutputParser

# --- 1. Pydantic Models for Structured Output ---
class GeneratedAnswer(BaseModel):
    """The generated answer based on the context."""
    answer: str = Field(description="The final answer to the user's query.")
    is_answerable: bool = Field(description="Is the query answerable based *only* on the provided context?")

class AnswerEvaluation(BaseModel):
    """A model to evaluate the quality of the generated answer based on the context."""
    confidence_score: conint(ge=0, le=100) = Field(description="A confidence score from 0 to 100 on how well the answer is grounded, relevant, and complete based on the context.")
    explanation: str = Field(description="A brief explanation for the assigned confidence score.")

# --- 2. Backend Functions (Separated for Caching) ---
def build_knowledge_base(files: list, progress=gr.Progress(track_tqdm=True)):
    """
    ONE-TIME SETUP FUNCTION: Takes uploaded files and builds the vector store retriever.
    """
    if not files:
        raise gr.Error("Please upload at least one document.")

    progress(0.1, desc="Preparing documents...")
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            shutil.copy(file.name, temp_dir)
        
        # --- THIS IS THE CORRECTED SECTION ---
        all_docs = []
        # Load PDF files
        pdf_loader = DirectoryLoader(temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
        all_docs.extend(pdf_loader.load())

        # Load TXT files
        txt_loader = DirectoryLoader(temp_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        all_docs.extend(txt_loader.load())
        # --- END OF CORRECTION ---

    if not all_docs:
        raise gr.Error("Error: Could not load content from any uploaded files.")

    progress(0.2, desc="Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_docs)

    progress(0.4, desc="Generating embeddings (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    progress(0.9, desc="Knowledge base is ready.")
    return vector_store.as_retriever(search_kwargs={"k": 5}), "✅ Knowledge base is built and ready to be queried."

def query_knowledge_base(retriever, query: str, progress=gr.Progress(track_tqdm=True)):
    """
    FAST QUERY FUNCTION: Takes the pre-built retriever and a query to generate an answer.
    """
    if retriever is None:
        raise gr.Error("Please build the knowledge base first by uploading documents and clicking the 'Build' button.")
    if not query.strip():
        raise gr.Error("Please enter a query.")
        
    api_key = userdata.get('GOOGLE_API_KEY')
    if not api_key:
        raise gr.Error("GOOGLE_API_KEY not found in Colab Secrets.")

    progress(0.5, desc="Retrieving relevant context...")
    context = "\n---\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.0)

    if not context.strip():
        generated_answer = {"answer": "I could not find relevant information in the provided documents to answer this question.", "is_answerable": False}
        evaluation = {"confidence_score": 0, "explanation": "No context was found in the documents to support any answer."}
    else:
        progress(0.7, desc="Generating answer...")
        generation_parser = JsonOutputParser(pydantic_object=GeneratedAnswer)
        generation_prompt = ChatPromptTemplate.from_template("Based ONLY on the context below, provide a direct answer. If the context is insufficient, state that the query is not answerable.\n{format_instructions}\n\n## Context\n{context}\n\n## Query\n{query}", partial_variables={"format_instructions": generation_parser.get_format_instructions()})
        generation_chain = generation_prompt | llm | generation_parser
        generated_answer = generation_chain.invoke({"context": context, "query": query})

        progress(0.9, desc="Performing self-evaluation...")
        evaluation_parser = JsonOutputParser(pydantic_object=AnswerEvaluation)
        evaluation_prompt = ChatPromptTemplate.from_template("You are an impartial judge. Evaluate the quality of the 'Generated Answer' based *only* on the 'Provided Context'. Score from 0 (not at all supported) to 100 (fully supported and relevant). Explain your reasoning.\n{format_instructions}\n\n## Provided Context\n{context}\n\n## User Query\n{query}\n\n## Generated Answer\n{answer}", partial_variables={"format_instructions": evaluation_parser.get_format_instructions()})
        evaluation_chain = evaluation_prompt | llm | evaluation_parser
        evaluation = evaluation_chain.invoke({"context": context, "query": query, "answer": generated_answer['answer']})

    # Format outputs for the UI
    score = evaluation['confidence_score']
    score_color = '#4CAF50' if score >= 80 else '#FFC107' if score >= 50 else '#F44336'
    score_md = f"""<div class="output-card"><div class="card-header">🎯 AUTOMATED CONFIDENCE SCORE</div><div class="score-value" style="color: {score_color};">{score}%</div><div class="score-explanation"><strong>AI Justification:</strong> {evaluation['explanation']}</div></div>"""
    generated_md = generated_answer['answer']
    retrieved_md = f"```\n{context}\n```" if context.strip() else "No context was retrieved."

    progress(1, desc="Done.")
    return score_md, generated_md, retrieved_md

# --- 3. Build the Professional Gradio UI ---
css = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { background: #1A1C21; color: #E0E0E0; }
.gr-panel { background-color: #242930 !important; border: none !important; border-radius: 12px !important; }
.gr-box { background-color: #242930 !important; border: none !important; }
.gr-accordion { background-color: #2B303A !important; border: 1px solid #3F4652 !important; border-radius: 8px !important; }
.output-card { padding: 20px; border-radius: 12px; background: #2B303A; margin-bottom: 20px; border: 1px solid #3F4652; }
.card-header { font-size: 14px; font-weight: bold; color: #9E9E9E; margin-bottom: 15px; letter-spacing: 1px; text-transform: uppercase; }
.score-value { font-size: 52px; font-weight: bold; text-align: center; }
.score-explanation { font-size: 14px; color: #B0B0B0; margin-top: 15px; text-align: center; }
"""
with gr.Blocks(theme=gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "sans-serif"]), css=css) as demo:
    # State variable to hold the retriever object between runs. This is our cache.
    retriever_state = gr.State()

    gr.Markdown("# 🤖 Document Compliance Tool")
    gr.Markdown("A two-step tool to get answers from your documents. First, build the knowledge base, then ask questions.")

    with gr.Row(variant='panel'):
        # --- LEFT COLUMN (Inputs & Controls) ---
        with gr.Column(scale=1):
            gr.Markdown("### **STEP 1: Build Knowledge Base**")
            files_input = gr.File(label="Upload Documents", file_count="multiple", file_types=[".pdf", ".txt"])
            build_button = gr.Button("Build Knowledge Base", variant="primary")
            build_status = gr.Markdown("Status: Waiting for documents...")
            
            gr.Markdown("---")
            
            gr.Markdown("### **STEP 2: Ask a Question**")
            query_input = gr.Textbox(label="Question", placeholder="e.g., What were the key findings of the report?", lines=4, interactive=True)
            run_button = gr.Button("Get Answer", variant="primary")

        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=2):
            gr.Markdown("### **RESULTS**")
            score_output = gr.Markdown(label="Confidence Score")
            with gr.Accordion("💡 Generated Answer", open=True):
                generated_output = gr.Markdown()
            with gr.Accordion("📚 Retrieved Context (Evidence)", open=False):
                retrieved_output = gr.Markdown()

    # --- Button Click Events ---
    build_button.click(
        fn=build_knowledge_base,
        inputs=[files_input],
        outputs=[retriever_state, build_status] # Store the retriever in the state
    )

    run_button.click(
        fn=query_knowledge_base,
        inputs=[retriever_state, query_input], # Use the retriever from the state
        outputs=[score_output, generated_output, retrieved_output]
    )

demo.launch(share=True, debug=True)
