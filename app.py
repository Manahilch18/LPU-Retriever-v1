import os
import shutil
import gradio as gr
from dotenv import load_dotenv

# 1. CHAIN LOGIC (2026 Modular Paths)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
GROQ_MODELS = ["llama-3.3-70b-versatile"]

CUSTOM_CSS = """
body, .gradio-container { background-color: #0a0a0b !important; color: #e5e7eb !important; }
#header-info { border-bottom: 1px solid #1f2937; margin-bottom: 2rem; padding-bottom: 1rem; }
.status-pill { font-family: monospace; font-size: 11px; background: #111113; border: 1px solid #1f2937; padding: 4px 12px; border-radius: 6px; display: inline-flex; align-items: center; gap: 8px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.bg-green { background-color: #10b981; }
.bg-blue { background-color: #3b82f6; }
"""

class RAGBot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.vector_store = None
        
    def get_llm(self, model_name):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY missing.")
        return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.1, streaming=True)

    def ingest_documents(self, data_folder=DATA_PATH):
        if not os.path.exists(data_folder): os.makedirs(data_folder)
        loaders = {
            ".pdf": DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader),
        }
        docs = []
        for loader in loaders.values(): docs.extend(loader.load())
        if not docs: return "No documents found."
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(VECTOR_DB_PATH)
        return f"Indexed {len(docs)} documents."

    def load_vector_store(self):
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            return True
        return False

    def search_documents(self, query):
        if not self.vector_store and not self.load_vector_store(): return "DB Empty."
        results = self.vector_store.similarity_search(query, k=3)
        return "## 🔍 Context\n\n" + "\n\n".join([f"> {res.page_content}" for res in results])

    def chat_stream(self, message, history, model_name):
        if not self.vector_store and not self.load_vector_store():
            yield "Please upload documents first."
            return
        try:
            llm = self.get_llm(model_name)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Context:\n{context}"),
                ("human", "{input}"),
            ])
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
            full_answer = ""
            for chunk in chain.stream({"input": message}):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    yield full_answer
        except Exception as e: yield f"⚠️ Error: {str(e)}"

bot = RAGBot()

def handle_upload(files):
    if not files: return "No files."
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
    for file in files: shutil.copy(file.name, os.path.join(DATA_PATH, os.path.basename(file.name)))
    return bot.ingest_documents()

# --- UI START ---
with gr.Blocks() as demo:
    with gr.Group(elem_id="header-info"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# RAG Engineering Blueprint")
                gr.Markdown("PRODUCTION-GRADE V1.1.0")
            with gr.Column(scale=1):
                gr.HTML('<div class="status-pill"><div class="status-dot bg-green"></div>GROQ: ACTIVE</div>')

    with gr.Tabs():
        with gr.Tab("🧠 Discovery Hub"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.ChatInterface(fn=bot.chat_stream, additional_inputs=[gr.Dropdown(choices=GROQ_MODELS, value=GROQ_MODELS[0])])
                with gr.Column(scale=1):
                    search_input = gr.Textbox(label="Search")
                    search_btn = gr.Button("Search")
                    search_display = gr.Markdown()
                    search_btn.click(bot.search_documents, inputs=[search_input], outputs=[search_display])

        with gr.Tab("🛠️ System Management"):
            fu = gr.File(file_count="multiple")
            pb = gr.Button("Ingest", variant="primary")
            st = gr.Textbox(label="Status")
            pb.click(handle_upload, inputs=[fu], outputs=[st])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="blue"), css=CUSTOM_CSS)