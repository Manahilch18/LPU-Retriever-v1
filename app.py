import os
import shutil
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
GROQ_MODELS = [
    "llama-3.3-70b-versatile"
    
     
]

# Elegant Dark CSS matching the design requirements
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0a0a0b !important;
    color: #e5e7eb !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}
.dark .bg-gray-50, .dark .bg-gray-100, .dark .bg-white {
    background-color: #111113 !important;
}
.border-gray-200, .border {
    border-color: #1f2937 !important;
}
h1 {
    color: #ffffff !important;
    letter-spacing: -0.025em !important;
    font-weight: 700 !important;
}
.text-gray-500 {
    color: #9ca3af !important;
}
#header-info {
    border-bottom: 1px solid #1f2937;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
}
.status-pill {
    font-family: monospace;
    font-size: 11px;
    background: #111113;
    border: 1px solid #1f2937;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.bg-green { background-color: #10b981; }
.bg-blue { background-color: #3b82f6; }
"""

class RAGBot:
    def __init__(self):
        # Explicitly set cache folder for persistence in some environments
        self.embeddings = HuggingFaceEmbeddings(model_name=sentence-transformers/all-MiniLM-L6-v2)
        self.vector_store = None
        
    def get_llm(self, model_name):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.1,
            streaming=True
        )

    def ingest_documents(self, data_folder=DATA_PATH):
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            return "Data folder initialized."

        loaders = {
            ".pdf": DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader),
        }
        
        docs = []
        for ext, loader in loaders.items():
            docs.extend(loader.load())

        if not docs:
            return "No documents found to process."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(VECTOR_DB_PATH)
        return f"Successfully indexed {len(docs)} documents ({len(splits)} chunks)."

    def load_vector_store(self):
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
            return True
        return False

    def search_documents(self, query):
        if not self.vector_store and not self.load_vector_store():
            return "Knowledge base empty. Please upload documents."
        
        results = self.vector_store.similarity_search(query, k=3)
        if not results: return "No relevant snippets found."
        
        output = "## 🔍 Context Research Results\n\n"
        for i, res in enumerate(results):
            source = os.path.basename(res.metadata.get('source', 'Unknown'))
            output += f"### {i+1}. Source: `{source}`\n> {res.page_content}\n\n---\n\n"
        return output

    def chat_stream(self, message, history, model_name):
        if not self.vector_store and not self.load_vector_store():
            yield "I don't have enough information yet. Please upload documents in the 'System Management' tab."
            return

        try:
            llm = self.get_llm(model_name)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

            system_prompt = (
                "You are an expert RAG Assistant. Use the provided context to answer questions precisely. "
                "Maintain a professional, technical tone. If the context doesn't contain the answer, "
                "state that clearly.\n\nContext:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Chain preparation
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            full_answer = ""
            for chunk in retrieval_chain.stream({"input": message}):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    yield full_answer
        except Exception as e:
            yield f"⚠️ Stream Error: {str(e)}"

# --- UI Interface ---
bot = RAGBot()

def handle_upload(files):
    if not files: return "No files provided."
    os.makedirs(DATA_PATH, exist_ok=True)
    for file in files:
        shutil.copy(file.name, os.path.join(DATA_PATH, os.path.basename(file.name)))
    return bot.ingest_documents()

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray"), css=CUSTOM_CSS) as demo:
    with gr.Div(elem_id="header-info"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# RAG Engineering Blueprint")
                gr.Markdown("PRODUCTION-GRADE IMPLEMENTATION GUIDE V1.1.0", elem_classes=["text-gray-500"])
            with gr.Column(scale=1, variant="compact"):
                gr.HTML('<div class="status-pill"><div class="status-dot bg-green"></div>GROQ_API: ACTIVE</div>')
                gr.HTML('<div class="status-pill"><div class="status-dot bg-blue"></div>HUGGINGFACE: SYNCED</div>')

    with gr.Tabs():
        with gr.Tab("🧠 Discovery Hub"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.ChatInterface(
                        fn=bot.chat_stream,
                        additional_inputs=[
                            gr.Dropdown(choices=GROQ_MODELS, value=GROQ_MODELS[0], label="Engine Architecture")
                        ],
                        stop_btn="Halt Generation"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Semantic Lookup")
                    search_input = gr.Textbox(label="Query Documents", placeholder="Find direct snippets...")
                    search_btn = gr.Button("Search Knowledge", variant="secondary")
                    search_display = gr.Markdown("*Research results will appear here...*")
                    search_btn.click(bot.search_documents, inputs=[search_input], outputs=[search_display])

        with gr.Tab("🛠️ System Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Data Pipeline")
                    fu = gr.File(file_count="multiple", label="Ingest PDF/TXT Records")
                    pb = gr.Button("Execute Ingestion", variant="primary")
                    st = gr.Textbox(label="Pipeline Status", interactive=False)
                    pb.click(handle_upload, inputs=[fu], outputs=[st])
                
                with gr.Column():
                    gr.Markdown("### System Architecture")
                    gr.Markdown("""
                    - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
                    - **Vector Engine**: FAISS (Facebook AI Similarity Search)
                    - **LLM Provider**: Groq LPU™ Inference
                    - **Persistence**: Local serialization in `/vectorstore`
                    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3000)
