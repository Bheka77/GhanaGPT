# GhanaGPT 🇬🇭
An intelligent AI assistant with document analysis, web search capabilities, and support for Ghanaian languages (Ga and Twi). Built with Streamlit, LangChain, and LangGraph.

# 🌟 Features

* 💬 Smart Conversational AI: Powered by Groq's LLM with context-aware responses
* 📄 Document RAG: Upload and query PDFs, Word documents, and text files
* 🌐 Web Search Integration: Real-time information retrieval using DuckDuckGo
* 🔄 Hybrid Mode: Combines document knowledge with web search for comprehensive answers
* 🌍 Local Language Support: Full translation support for Ga and Twi languages
* 🧠 Intelligent Memory Management: Automatic conversation summarization for extended chats
* 🎨 Modern Dark UI: Beautiful, responsive interface with dark theme
* 💾 Persistent Storage: Chat history saved per language and session

# 🚀 Quick Start
## Prerequisites
*Python 3.8 or higher
*Groq API key
*Ghana NLP API key (for language translation)

## Installation
* Clone the repository
git clone https://github.com/yourusername/ghanagpt.git
cd ghanagpt

* Install dependencies
pip install -r requirements.txt

* Set up environment variables
Create a .env file in the root directory:
envGROQ_API_KEY=your_groq_api_key_here
GHANA_NLP_KEY=your_ghana_nlp_key_here

* Run the application
streamlit run web.py
