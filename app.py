from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os, random
import gradio as gr

# ================== Config ==================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATA_PATH = os.environ.get("DATA_PATH", "./data")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
PORT = int(os.environ.get("PORT", 5000))

# 🎨 Color themes
color_themes = [
    {"bg": "#0d1117", "chat_bg": "#161b22", "user": "#00ff7f", "bot": "#00bfff"},
    {"bg": "#1e1e2f", "chat_bg": "#2b2d42", "user": "#ff7f50", "bot": "#40e0d0"},
    {"bg": "#0f0f0f", "chat_bg": "#1a1a1a", "user": "#ffd700", "bot": "#00fa9a"},
    {"bg": "#101820", "chat_bg": "#1c1c28", "user": "#ff69b4", "bot": "#87cefa"},
    {"bg": "#0b0c10", "chat_bg": "#1f2833", "user": "#66fcf1", "bot": "#45a29e"}
]
theme = random.choice(color_themes)

def get_custom_css(theme):
    return f"""
    body {{
        background-color: {theme['bg']};
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }}
    #chatbot {{
        border: 2px solid #30363d;
        border-radius: 12px;
        padding: 10px;
        background: {theme['chat_bg']};
        box-shadow: 0 0 20px {theme['user']}33;
    }}
    .message.user {{
        color: {theme['user']} !important;
        font-weight: bold;
    }}
    .message.bot {{
        color: {theme['bot']} !important;
        font-style: italic;
    }}
    footer {{
        text-align: center;
        color: #888;
        font-size: 14px;
        padding-top: 10px;
    }}
    """

# ================== LangChain Setup ==================
def initialize_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        raise FileNotFoundError(f"No PDFs found! Upload files into {DATA_PATH}")
    
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    PROMPT = PromptTemplate(
        template="""You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
{context}
User: {question}
Chatbot:""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

# ================== Initialization ==================
llm = initialize_llm()
if not os.path.exists(CHROMA_DB_PATH):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# ================== Chatbot Function ==================
def chatbot_response(message, history):
    if not message.strip():
        return "⚠️ Please provide a valid input"
    try:
        return qa_chain.run(message)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ================== Gradio UI ==================
custom_css = get_custom_css(theme)
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"<h1 style='color:{theme['user']};text-align:center;'>🧠 Mental Health Chatbot 🤖</h1>")
    gr.Markdown("<p style='text-align:center;color:#9e9e9e;'>A compassionate chatbot for mental well-being.<br>For serious concerns, contact a professional.</p>")

    chatbot_ui = gr.ChatInterface(
        fn=chatbot_response,
        title="Dynamic Themed Mental Health Bot",
        description="💬 Chat with a supportive assistant",
        chatbot=gr.Chatbot(elem_id="chatbot", height=500)
    )

    def change_theme():
        new_theme = random.choice(color_themes)
        return gr.update(css=get_custom_css(new_theme))

    change_theme_btn = gr.Button("🎨 Change Theme")
    change_theme_btn.click(fn=change_theme, outputs=[])

    gr.Markdown("<footer>💡 General support only. Seek help from licensed professionals for urgent issues.</footer>")

# ================== Launch ==================
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT, share=False)
