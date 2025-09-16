import time
import random
import streamlit as st
from llm import LLM as llm
from storage import Storage as storage
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="GhanaGPT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Enhanced CSS with RAG/Search Features
# -------------------------
st.markdown(
    """
    <style>
      /* Import Inter font */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      
      /* CSS Variables */
      :root {
        --bg: #0B1220;
        --panel: #0F172A;
        --panel-2: #111827;
        --text: #E5E7EB;
        --muted: #9CA3AF;
        --primary: #6C5CE7;
        --accent: #8B5CF6;
        --success: #22C55E;
        --warning: #F59E0B;
        --info: #3B82F6;
        --radius: 16px;
        --shadow: 0 8px 25px rgba(0,0,0,.3);
      }

      /* Global styling */
      html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Inter', system-ui, sans-serif;
      }

      /* Main background */
      [data-testid="stAppViewContainer"] { 
        background: var(--bg);
      }
      
      /* Container */
      .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        max-width: 1000px;
      }

      /* Hide Streamlit branding */
      #MainMenu, footer, header { visibility: hidden; }

      /* Sidebar styling */
      [data-testid="stSidebar"] { 
        background: var(--panel); 
        border-right: 1px solid rgba(255,255,255,.1);
      }
      
      [data-testid="stSidebar"] h3 {
        color: var(--text);
        font-weight: 600;
      }
      
      [data-testid="stSidebar"] .stCaption { 
        color: var(--muted); 
        line-height: 1.4;
      }

      /* Mode selector buttons */
      .mode-selector {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 0.5rem;
        background: rgba(255,255,255,.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.1);
      }
      
      .mode-button {
        flex: 1;
        padding: 0.75rem;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: var(--muted);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
      }
      
      .mode-button.active {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white;
        border-color: var(--accent);
        box-shadow: 0 4px 12px rgba(108,92,231,.3);
      }
      
      .mode-button:hover:not(.active) {
        background: rgba(255,255,255,.1);
        border-color: rgba(255,255,255,.2);
      }
      
      .mode-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
      }
      
      .mode-chat { background: rgba(139,92,246,.2); color: var(--accent); }
      .mode-rag { background: rgba(34,197,94,.2); color: var(--success); }
      .mode-web { background: rgba(59,130,246,.2); color: var(--info); }
      .mode-hybrid { background: rgba(245,158,11,.2); color: var(--warning); }

      /* File uploader styling */
      [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.05);
        border: 2px dashed rgba(255,255,255,.2);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
      }
      
      [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: rgba(108,92,231,.1);
      }

      /* Document info cards */
      .doc-info-card {
        background: rgba(34,197,94,.1);
        border: 1px solid rgba(34,197,94,.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: var(--text);
      }
      
      .doc-info-card .doc-name {
        font-weight: 600;
        color: var(--success);
      }
      
      .doc-info-card .doc-meta {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 0.25rem;
      }

      /* Welcome section */
      .welcome-container {
        background: linear-gradient(135deg, rgba(108,92,231,.15), rgba(139,92,246,.08));
        border: 1px solid rgba(108,92,231,.25);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: var(--shadow);
      }
      
      .welcome-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
      }
      
      .welcome-subtitle {
        color: var(--muted);
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
      }
      
      .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
      }
      
      .feature-card {
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: left;
        transition: all 0.3s ease;
      }
      
      .feature-card:hover {
        background: rgba(255,255,255,.08);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,.3);
      }
      
      .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
      }
      
      .feature-title {
        color: var(--text);
        font-weight: 600;
        margin-bottom: 0.25rem;
      }
      
      .feature-desc {
        color: var(--muted);
        font-size: 0.85rem;
      }

      /* Chat messages */
      div[data-testid="stChatMessage"] { 
        background: rgba(255,255,255,.05); 
        border: 1px solid rgba(255,255,255,.1); 
        border-radius: 18px; 
        padding: 1rem 1.25rem; 
        margin: 1rem 0; 
        box-shadow: 0 4px 15px rgba(0,0,0,.2);
        animation: fadeIn .3s ease-out;
      }

      /* Context-aware message styling */
      .message-with-context {
        border-left: 3px solid var(--success);
      }
      
      .message-with-web {
        border-left: 3px solid var(--info);
      }
      
      .message-with-hybrid {
        border-left: 3px solid var(--warning);
      }

      /* TTS Button styling */
      [data-testid="column"]:last-child .stButton button {
        background: rgba(34,197,94,.15) !important;
        border: 1px solid rgba(34,197,94,.3) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        display: flex !important;
        flex: 1 !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 18px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
      }

      [data-testid="column"]:last-child .stButton button:hover {
        background: rgba(34,197,94,.25) !important;
        transform: scale(1.1) !important;
        box-shadow: 0 4px 12px rgba(34,197,94,.3) !important;
        border-color: rgba(34,197,94,.4) !important;
      }

      /* Typing cursor animation */
      @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
      }
      
      @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(10px); } 
        to { opacity: 1; transform: translateY(0); } 
      }

      .typing-cursor {
        animation: blink 1s infinite;
        color: var(--accent);
        font-weight: bold;
      }

      /* Chat input */
      div[data-testid="stChatInput"] > div { 
        background: var(--panel-2) !important; 
        border: 1px solid rgba(255,255,255,.15) !important; 
        border-radius: 24px !important; 
        box-shadow: var(--shadow) !important;
      }
      
      div[data-testid="stChatInput"] textarea { 
        background: transparent !important; 
        color: var(--text) !important; 
        font-size: 1rem !important;
      }

      /* Context info styling */
      .context-info {
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: var(--muted);
      }
      
      .context-info-item {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
      }
      
      .context-info-label {
        color: var(--muted);
      }
      
      .context-info-value {
        color: var(--text);
        font-weight: 500;
      }

      /* Scrollbar */
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: var(--panel); }
      ::-webkit-scrollbar-thumb { 
        background: rgba(255,255,255,.2); 
        border-radius: 3px; 
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Language Configuration
# -------------------------
AVAILABLE_LANGUAGES = {
    "en": "English",
    "gaa": "Ga",
    "tw": "Twi"
}

# -------------------------
# Session State Initialization
# -------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(random.randint(100, 999))

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": st.session_state.chat_id}}

if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = llm()

if "context_info" not in st.session_state:
    st.session_state.context_info = {
        "total_messages": 0, 
        "summaries_created": 0, 
        "context_status": "Optimal",
        "documents_loaded": False,
        "num_documents": 0
    }

if "context_mode" not in st.session_state:
    st.session_state.context_mode = "chat"

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


# -------------------------
# Helper Functions
# -------------------------
def get_mode_indicator(mode):
    """Get the visual indicator for the current mode"""
    indicators = {
        "chat": ("<span class='mode-indicator mode-chat'>💬 Chat</span>", "Normal chat mode"),
        "rag": ("<span class='mode-indicator mode-rag'>📄 Documents</span>", "Using uploaded documents"),
        "web_search": ("<span class='mode-indicator mode-web'>🌐 Web</span>", "Web Search"),
        "hybrid": ("<span class='mode-indicator mode-hybrid'>🔀 Hybrid</span>", "Documents + Web")
    }
    return indicators.get(mode, ("", ""))

# -------------------------
# Sidebar with Enhanced Features
# -------------------------
with st.sidebar:
    st.markdown("### 🤖 Model Options")
    st.caption("Choose a language")
    
    selected_language = st.selectbox(
        "Languages",
        options=list(AVAILABLE_LANGUAGES.keys()),
        format_func=lambda x: AVAILABLE_LANGUAGES[x],
        index=list(AVAILABLE_LANGUAGES.keys()).index(st.session_state.selected_language),
        key="language_selector",
    )

    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.rerun()
    
    st.markdown("---")
    
    # Context Mode Selector
    st.markdown("### 🎯 Chat Mode")
    
    # Show current mode description
    mode_indicator, mode_desc = get_mode_indicator(st.session_state.context_mode)
    if mode_indicator:
        st.markdown(f"<div style='text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;'>{mode_indicator}</div>", 
                   unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True, 
                    type="primary" if st.session_state.context_mode == "chat" else "secondary"):
            st.session_state.context_mode = "chat"
            st.rerun()
        
        if st.button("📄 Documents", use_container_width=True, 
                    type="primary" if st.session_state.context_mode == "rag" else "secondary",
                    disabled=not st.session_state.llm_instance.documents_loaded):
            st.session_state.context_mode = "rag"
            st.rerun()
    
    with col2:
        if st.button("🌐 Web Search", use_container_width=True,
                    type="primary" if st.session_state.context_mode == "web_search" else "secondary"):
            st.session_state.context_mode = "web_search"
            st.rerun()

        if st.button("🔀 Hybrid", use_container_width=True,
                    type="primary" if st.session_state.context_mode == "hybrid" else "secondary",
                    disabled=not st.session_state.llm_instance.documents_loaded):
            st.session_state.context_mode = "hybrid"
            st.rerun()
    
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📁 Document Upload")
    st.caption("Upload documents for RAG")
    
    uploaded_files = st.file_uploader(
        "Upload file(s)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'doc'],
        key="doc_uploader",
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_files:
        if st.button("📥 Load Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                result = st.session_state.llm_instance.load_documents(uploaded_files)
                
                if result['success']:
                    st.success(result['message'])
                    st.session_state.uploaded_docs = result['metadata']
                    st.session_state.context_info.update({
                        "documents_loaded": True,
                        "num_documents": len(result['metadata'])
                    })
                else:
                    st.error(result['message'])
            st.rerun()
    
    # Display loaded documents
    if st.session_state.uploaded_docs:
        st.markdown("#### 📚 Loaded Documents")
        for doc in st.session_state.uploaded_docs:
            st.markdown(
                f"""<div class='doc-info-card'>
                    <div class='doc-name'>📄 {doc['name']}</div>
                    <div class='doc-meta'>Type: {doc['type'].upper()} | Pages: {doc['num_pages']}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        if st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state.llm_instance.vector_store = None
            st.session_state.llm_instance.documents_loaded = False
            st.session_state.llm_instance.document_metadata = []
            st.session_state.uploaded_docs = []
            st.session_state.context_info.update({
                "documents_loaded": False,
                "num_documents": 0
            })
            if st.session_state.context_mode in ["rag", "hybrid"]:
                st.session_state.context_mode = "chat"
            st.rerun()
    
    st.markdown("---")
    
    # Context Information
    st.markdown("### 🧠 Memory Status")
    info = st.session_state.context_info
    
    st.markdown(
        f"""<div class='context-info'>
            <div class='context-info-item'>
                <span class='context-info-label'>📊 Messages:</span>
                <span class='context-info-value'>{info['total_messages']}</span>
            </div>
            <div class='context-info-item'>
                <span class='context-info-label'>📄 Summaries:</span>
                <span class='context-info-value'>{info['summaries_created']}</span>
            </div>
            <div class='context-info-item'>
                <span class='context-info-label'>⚡ Status:</span>
                <span class='context-info-value'>{info['context_status']}</span>
            </div>
            <div class='context-info-item'>
                <span class='context-info-label'>📄 Documents:</span>
                <span class='context-info-value'>{info['num_documents']}</span>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Chat Controls
    st.markdown("### 💬 Chat Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_info.update({
            "total_messages": 0,
            "summaries_created": 0,
            "context_status": "Optimal"
        })
        st.rerun()
    
    st.markdown("---")
    
    # Session Info
    st.markdown("### 💡 Info")
    st.caption(f"• Session ID: **{st.session_state.chat_id}**")
    st.caption(f"• Language: **{AVAILABLE_LANGUAGES[st.session_state.selected_language]}**")
    st.caption(f"• Mode: **{st.session_state.context_mode.replace('_', ' ').title()}**")

# -------------------------
# Main Chat Interface
# -------------------------

# Welcome message when chat is empty
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class='welcome-container'>
            <div class='welcome-title'>GhanaGPT 💬</div>
            <div class='welcome-subtitle'>
                Your AI assistant with document analysis and web search capabilities.<br>
                Upload documents, search the web, or just chat! Available in English and Ghanaian languages.
            </div>
            <div class='feature-grid'>
                <div class='feature-card'>
                    <div class='feature-icon'>💬</div>
                    <div class='feature-title'>Smart Chat</div>
                    <div class='feature-desc'>Engage in natural conversations with context awareness.</div>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>📄</div>
                    <div class='feature-title'>Document RAG</div>
                    <div class='feature-desc'>Upload and query your documents intelligently.</div>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🌐</div>
                    <div class='feature-title'>Web Search</div>
                    <div class='feature-desc'>Get real-time information from the web.</div>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🔀</div>
                    <div class='feature-title'>Hybrid Mode</div>
                    <div class='feature-desc'>Combine documents and web for best results.</div>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🌍</div>
                    <div class='feature-title'>Local Languages</div>
                    <div class='feature-desc'>Choose between Ga and Twi and experience AI in your local language.</div>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>⚡</div>
                    <div class='feature-title'>Fast & Reliable</div>
                    <div class='feature-desc'>Optimized for quick responses and minimal downtime.</div>
                </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display chat messages with cached translations
for i, message in enumerate(st.session_state.messages):
    avatar = "🤖" if message["role"] == "ai" else "👤"
    role = "assistant" if message["role"] == "ai" else message["role"]
    
    # For English, use original content directly
    if st.session_state.selected_language == "en":
        text = message["content"]
    else:
        # Check if translation already exists for current language
        lang_key = f"content_{st.session_state.selected_language}"
        
        if lang_key in message:
            # Use cached translation
            text = message[lang_key]
        else:
            # Translate and store for future use
            text = st.session_state.llm_instance.translate(
                lan_code=st.session_state.selected_language, 
                text=message["content"]
            )
            # Store the translation in the message
            message[lang_key] = text
    
    # Add mode indicator for AI responses
    message_class = ""
    if message["role"] == "ai" and "mode" in message:
        if message["mode"] == "rag":
            message_class = "message-with-context"
        elif message["mode"] == "web_search":
            message_class = "message-with-web"
        elif message["mode"] == "hybrid":
            message_class = "message-with-hybrid"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

# Chat input with context-aware processing
if prompt := st.chat_input("Ask anything"):
    if st.session_state.selected_language == "en":
        display_text = prompt
    else:
        display_text = st.session_state.llm_instance.translate(
            lan_code=st.session_state.selected_language, text=prompt)
    
    # Add user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(display_text)
    
    # Store message with both original and translated content
    user_message = {
        "role": "user", 
        "content": prompt,
    }
    
    # Only add translation if not English
    if st.session_state.selected_language != "en":
        user_message[f"content_{st.session_state.selected_language}"] = display_text
    
    st.session_state.messages.append(user_message)
    
    # Generate AI response
    try:
        with st.chat_message("assistant", avatar="🤖"):
            mode_indicator, _ = get_mode_indicator(st.session_state.context_mode)
            if st.session_state.context_mode != "chat":
                st.markdown(mode_indicator, unsafe_allow_html=True)
            
            message_placeholder = st.empty()
            full_response = ""
            
            # Build langchain messages from history
            langchain_messages = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            langchain_messages.append(HumanMessage(content=prompt))
            
            # Show appropriate loading message based on mode
            loading_messages = {
                "chat": "...",
                "rag": "Searching documents...",
                "web_search": "Searching the web...",
                "hybrid": "Searching documents and web..."
            }
            
            with st.spinner(loading_messages.get(st.session_state.context_mode, "Processing...")):
                try:
                    state = {
                        "messages": langchain_messages,
                        "context_mode": st.session_state.context_mode
                    }
                    ai_response = st.session_state.llm_instance.graph().invoke(
                        state,
                        config=st.session_state.config
                    )
                    full_response = ai_response["messages"][-1].content if hasattr(
                        ai_response["messages"][-1], 'content'
                    ) else str(ai_response["messages"][-1])
                    
                    if st.session_state.selected_language == "en":
                        display_response = full_response
                    else:
                        display_response = st.session_state.llm_instance.translate(
                            lan_code=st.session_state.selected_language, text=full_response
                        )

                    streamed_text = ""
                    for char in display_response:
                        streamed_text += char
                        message_placeholder.markdown(
                            streamed_text + "<span class='typing-cursor'>▌</span>", 
                            unsafe_allow_html=True
                        )
                        time.sleep(0.03)
                    
                except Exception as e:
                    full_response = f"I apologize, but I encountered an error: {str(e)}"
                    display_response = full_response
            
            message_placeholder.markdown(display_response)
        
        # Save AI message with translation
        ai_message = {
            "role": "ai",
            "content": full_response,
            "mode": st.session_state.context_mode
        }
        
        if st.session_state.selected_language != "en":
            ai_message[f"content_{st.session_state.selected_language}"] = display_response
        
        st.session_state.messages.append(ai_message)
        
        # Update context info
        st.session_state.context_info["total_messages"] = len(st.session_state.messages)
        
        # Save to storage with language
        try:
            if len(st.session_state.messages) >= 2:
                storage_instance = storage(id=st.session_state.chat_id)
                # Save in the current language
                storage_instance.save_chat_history(
                    st.session_state.messages[-2:], 
                    st.session_state.selected_language
                )
        except Exception as e:
            print(f"Error saving to storage: {e}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")