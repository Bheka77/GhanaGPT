import os
import tempfile
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredWordDocumentLoader

# load env variables
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    should_summarize: bool
    summary_count: int
    context_mode: str
    rag_context: str
    web_results: str
    uploaded_docs: list

class LLM:
    def __init__(self):
        self.api = os.getenv("GROQ_API_KEY")
        self.model_name = "openai/gpt-oss-120b"  
        self.max_context_messages = 20 
        self.keep_recent_messages = 8
        self.ghana_nlp_key = os.getenv("GHANA_NLP_KEY")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = None
        self.documents_loaded = False
        
        self.search_tool = DuckDuckGoSearchRun()
        
        self.document_metadata = []
        
        self.current_context_mode = "chat"
        
        self._compiled_graph = None
        self._build_graph()

    def _build_graph(self):
        """Build the conversation graph with memory management - called once"""
        workflow = StateGraph(state_schema=ChatState)
        
        workflow.add_node("check_summary", self.check_summarization)
        workflow.add_node("summarize", self.summarize_conversation)
        workflow.add_node("model", self.model_layer)
        
        workflow.add_edge(START, "check_summary")
        
        def should_summarize_edge(state):
            return "summarize" if state.get("should_summarize", False) else "model"
        
        workflow.add_conditional_edges(
            "check_summary",
            should_summarize_edge,
            {"summarize": "summarize", "model": "model"}
        )
        
        workflow.add_edge("summarize", "model")
        workflow.add_edge("model", END)
        
        # Temporary memory for graph
        memory = MemorySaver()
        self._compiled_graph = workflow.compile(checkpointer=memory)

    def model(self):
        return init_chat_model(
            self.model_name, 
            model_provider="groq", 
            api_key=self.api,
            temperature=0.6
        )

    def load_documents(self, uploaded_files) -> Dict[str, Any]:
        """Load and process uploaded documents for RAG"""
        try:
            all_documents = []
            self.document_metadata = []
            failed_files = []
            
            for uploaded_file in uploaded_files:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    documents = []
                    
                    if file_extension == 'pdf':
                        try:
                            loader = PyPDFLoader(tmp_path)
                            documents = loader.load()
                        except Exception as pdf_error:
                            print(f"Error loading PDF {uploaded_file.name}: {str(pdf_error)}")
                            failed_files.append(f"{uploaded_file.name} (PDF error)")
                            continue
                            
                    elif file_extension == 'txt':
                        try:
                            loader = TextLoader(tmp_path, encoding='utf-8')
                            documents = loader.load()
                        except UnicodeDecodeError:
                            try:
                                loader = TextLoader(tmp_path, encoding='latin-1')
                                documents = loader.load()
                            except Exception as txt_error:
                                print(f"Error loading text file {uploaded_file.name}: {str(txt_error)}")
                                failed_files.append(f"{uploaded_file.name} (encoding error)")
                                continue
                        except Exception as txt_error:
                            print(f"Error loading text file {uploaded_file.name}: {str(txt_error)}")
                            failed_files.append(f"{uploaded_file.name} (text error)")
                            continue
                            
                    elif file_extension in ['docx', 'doc']:
                        try:
                            loader = Docx2txtLoader(tmp_path)
                            documents = loader.load()
                        except ImportError:
                            try:
                                loader = UnstructuredWordDocumentLoader(tmp_path)
                                documents = loader.load()
                            except ImportError:
                                print(f"No Word document loader available for {uploaded_file.name}")
                                failed_files.append(f"{uploaded_file.name} (no Word loader)")
                                continue
                        except Exception as docx_error:
                            print(f"Error loading Word document {uploaded_file.name}: {str(docx_error)}")
                            failed_files.append(f"{uploaded_file.name} (Word error)")
                            continue
                    else:
                        print(f"Skipping unsupported file type: {file_extension}")
                        failed_files.append(f"{uploaded_file.name} (unsupported type)")
                        continue
                    
                    # Validate loaded documents
                    if not documents:
                        print(f"No content extracted from {uploaded_file.name}")
                        failed_files.append(f"{uploaded_file.name} (empty)")
                        continue
                    
                    # Filter out empty documents
                    valid_documents = []
                    for doc in documents:
                        if hasattr(doc, 'page_content') and doc.page_content.strip():
                            doc.metadata['source_file'] = uploaded_file.name
                            doc.metadata['file_type'] = file_extension
                            valid_documents.append(doc)
                    
                    if valid_documents:
                        all_documents.extend(valid_documents)
                        self.document_metadata.append({
                            'name': uploaded_file.name,
                            'type': file_extension,
                            'num_pages': len(valid_documents)
                        })
                        print(f"Successfully loaded {len(valid_documents)} pages from {uploaded_file.name}")
                    else:
                        print(f"No valid content in {uploaded_file.name}")
                        failed_files.append(f"{uploaded_file.name} (no valid content)")
                    
                except Exception as file_error:
                    print(f"Unexpected error processing {uploaded_file.name}: {str(file_error)}")
                    failed_files.append(f"{uploaded_file.name} (processing error)")
                    
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            
            # Create vector store if we have documents
            if all_documents:
                try:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    
                    splits = text_splitter.split_documents(all_documents)
                    
                    valid_splits = [split for split in splits if split.page_content.strip()]
                    
                    if not valid_splits:
                        return {
                            'success': False,
                            'message': 'Documents were loaded but no valid text chunks could be created'
                        }
                    
                    print(f"Created {len(valid_splits)} chunks from documents")
                    
                    # Create vector store
                    self.vector_store = Chroma.from_documents(valid_splits, self.embeddings)
                    self.documents_loaded = True
                    
                    success_message = f'Successfully loaded {len(all_documents)} documents with {len(valid_splits)} chunks'
                    if failed_files:
                        success_message += f'\nFailed to load: {", ".join(failed_files)}'
                    
                    return {
                        'success': True,
                        'message': success_message,
                        'metadata': self.document_metadata,
                        'failed_files': failed_files
                    }
                    
                except Exception as vectorization_error:
                    print(f"Error creating vector store: {str(vectorization_error)}")
                    return {
                        'success': False,
                        'message': f'Documents loaded but vectorization failed: {str(vectorization_error)}',
                        'failed_files': failed_files
                    }
            else:
                error_msg = 'No valid documents were loaded'
                if failed_files:
                    error_msg += f'. Failed files: {", ".join(failed_files)}'
                return {
                    'success': False,
                    'message': error_msg,
                    'failed_files': failed_files
                }
                
        except Exception as e:
            print(f"Critical error in load_documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Critical error loading documents: {str(e)}'
            }

    def search_documents(self, query: str, k: int = 5) -> str:
        """Search through loaded documents using RAG"""
        if not self.documents_loaded or not self.vector_store:
            return ""
        
        try:
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            if not relevant_docs:
                return ""
            
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                content = doc.page_content.strip()
                context_parts.append(f"[Source: {source}]\n{content}")
            
            return "\n\n---\n\n".join(context_parts)
        
        except Exception as e:
            print(f"Error in search_documents: {str(e)}")
            return ""

    def search_web(self, query: str) -> str:
        """Search the web for information"""
        try:
            results = self.search_tool.run(query)
            
            if results:
                return f"Web Search Results:\n\n{results}"
            else:
                return ""
                
        except Exception as e:
            print(f"Error in search_web: {str(e)}")
            return ""

    def check_summarization(self, state: ChatState):
        """Check if conversation needs summarization"""
        message_count = len(state["messages"])
        should_summarize = message_count > self.max_context_messages
        
        return {
            "should_summarize": should_summarize,
            "summary_count": state.get("summary_count", 0),
            "context_mode": state.get("context_mode", self.current_context_mode)
        }

    def summarize_conversation(self, state: ChatState):
        """Summarize older messages to maintain context while reducing token usage"""
        if not state.get("should_summarize", False):
            return state
        
        messages = state["messages"]
        
        recent_messages = messages[-self.keep_recent_messages:]
        
        messages_to_summarize = messages[:-self.keep_recent_messages]
        
        if len(messages_to_summarize) > 0:
            conversation_text = ""
            for msg in messages_to_summarize:
                if hasattr(msg, 'content'):
                    role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                    conversation_text += f"{role}: {msg.content}\n"
            
            summary_prompt = f"""Please provide a concise summary of this conversation, 
            capturing the key topics, decisions, and context that would be important for continuing the conversation:

            {conversation_text}

            Summary:"""
            
            try:
                summary_response = self.model().invoke([HumanMessage(content=summary_prompt)])
                summary_message = AIMessage(content=f"[Previous conversation summary: {summary_response.content}]")
                
                new_messages = [summary_message] + recent_messages
                
                return {
                    "messages": new_messages,
                    "should_summarize": False,
                    "summary_count": state.get("summary_count", 0) + 1,
                    "context_mode": state.get("context_mode", self.current_context_mode)
                }
            
            except Exception as e:
                print(f"Error during summarization: {e}")
                return {
                    "messages": recent_messages,
                    "should_summarize": False,
                    "summary_count": state.get("summary_count", 0),
                    "context_mode": state.get("context_mode", self.current_context_mode)
                }
        
        return state

    def model_layer(self, state: ChatState):
        """Generate model response with context from RAG or web search"""
        try:
            messages = state["messages"]

            context_mode = state.get("context_mode", self.current_context_mode)
            
            print(f"Model layer - Context mode: {context_mode}")
            
            if context_mode in ["rag", "web_search", "hybrid"]:
                last_message = messages[-1]
                query = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                context_parts = []
                
                if context_mode in ["rag", "hybrid"] and self.documents_loaded:
                    rag_context = self.search_documents(query)
                    if rag_context:
                        context_parts.append(f"## Document Context:\n{rag_context}")
                        print(f"Added RAG context: {len(rag_context)} chars")
                
                if context_mode in ["web_search", "hybrid"]:
                    web_results = self.search_web(query)
                    if web_results:
                        context_parts.append(f"## Web Search Results:\n{web_results}")
                        print(f"Added web context: {len(web_results)} chars")
                
                if context_parts:
                    enhanced_prompt = f"""Based on the following context, please answer the user's question.
                    
                        {chr(10).join(context_parts)}

                        User Question: {query}

                        Please provide a comprehensive answer based on the provided context. 
                        If the context doesn't contain relevant information, please state that clearly.
                        Lastly, make sure your responses are brief and to the point."""
                    
                    messages = messages[:-1] + [HumanMessage(content=enhanced_prompt)]
                    print("Enhanced prompt with context")
            
            res = self.model().invoke(messages)
            return {"messages": res, "context_mode": context_mode}
            
        except Exception as e:
            print(f"Error in model_layer: {str(e)}")
            error_message = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
            return {"messages": error_message, "context_mode": state.get("context_mode", "chat")}

    def graph(self):
        """Return the compiled graph (already built in __init__)"""
        return self._compiled_graph
    
    def get_context_info(self, state: ChatState):
        """Get information about current context usage"""
        info = {
            "total_messages": len(state.get("messages", [])),
            "summaries_created": state.get("summary_count", 0),
            "context_status": "Optimal" if len(state.get("messages", [])) <= self.max_context_messages else "Will summarize next",
            "documents_loaded": self.documents_loaded,
            "num_documents": len(self.document_metadata) if self.document_metadata else 0,
            "current_mode": self.current_context_mode
        }
        
        if self.vector_store:
            info["vector_store_size"] = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else "Unknown"
        
        return info

    def translate(self, lan_code: str, text: str) -> str:
        """
        Translate text to specified local language with simple error handling.
        """

        if lan_code == "en" or not text or not text.strip():
            return text
        
        code = f"en-{lan_code}"
        url = "https://translation-api.ghananlp.org/v1/translate"
        
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.ghana_nlp_key
        }
        
        payload = {
            "in": text,
            "lang": code
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return str(result)
            else:
                print(f"Translation API error: {response.status_code}")
                return text
                
        except Exception as e:
            print(f"Translation error: {e}")
            return text
