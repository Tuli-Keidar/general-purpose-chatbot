import streamlit as st
import os
import asyncio
import tiktoken
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Document processing libraries
import PyPDF2
import docx

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
    AzureTextEmbedding
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.memory import VolatileMemoryStore
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable


class DocumentProcessor:
    """Enhanced document processor with proper format support"""
    
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx', '.csv', '.xlsx'}
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def extract_text(self, file_path: str) -> str:
        """Extract text from various document formats"""
        ext = Path(file_path).suffix.lower()
        
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
            
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif ext == '.pdf':
                text = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
                
            elif ext == '.docx':
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                return f"CSV Data:\n{df.to_string()}"
                
            elif ext == '.xlsx':
                df = pd.read_excel(file_path)
                return f"Excel Data:\n{df.to_string()}"
                
        except Exception as e:
            raise ValueError(f"Error processing {ext} file: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean text for embedding processing"""
        import re
        if not isinstance(text, str):
            text = str(text)
        
        # Encode to UTF-8 bytes and decode, replacing errors
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove control characters and normalize whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Keep only printable ASCII and common Unicode characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Ensure text is not too short or empty
        text = text.strip()
        if len(text) < 5:  # Reduced minimum length
            return ""
            
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with token-aware splitting"""
        if not text.strip():
            return []
        
        # Clean text first
        text = self.clean_text(text)
        if not text:
            return []
            
        chunks = []
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunk_text = self.clean_text(chunk_text)  # Clean again after decoding
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)
            start = end - overlap
            
        return chunks


class EnhancedAIApp:
    """Enhanced AI application with proper RAG and error handling"""
    
    def __init__(self):
        # Load .env from current directory explicitly
        from pathlib import Path
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        
        self.document_processor = DocumentProcessor()
        self.initialize_session_state()
        self.setup_services()
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'chat_messages': [],
            'processed_documents': {},
            'document_chunks_count': 0,
            'total_tokens': 0,
            'estimated_cost': 0.0,
            'app_initialized': False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def validate_environment(self) -> bool:
        """Validate required environment variables"""
        required_vars = [
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_ENDPOINT", 
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            st.error(f"Missing environment variables: {', '.join(missing_vars)}")
            st.info("Please create a .env file with the required Azure OpenAI credentials.")
            return False
        return True
    
    def setup_services(self):
        """Setup kernel and services with proper error handling"""
        if not self.validate_environment():
            st.stop()
            
        try:
            # Initialize kernel
            self.kernel = Kernel()
            
            # Use endpoint as-is for Semantic Kernel (it handles cognitive services endpoints)
            clean_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            # Setup chat completion service
            chat_service = AzureChatCompletion(
                service_id="chat",
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                endpoint=clean_endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            self.kernel.add_service(chat_service)
            
            # Setup embedding service  
            embedding_service = AzureTextEmbedding(
                service_id="embedding",
                deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                endpoint=clean_endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            self.kernel.add_service(embedding_service)
            
            # Setup direct Azure OpenAI client for embeddings (bypass Semantic Kernel)
            from openai import AzureOpenAI
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01",
                azure_endpoint=clean_endpoint
            )
            
            # Document storage with embeddings
            self.document_store = {}
            self.embedding_service = embedding_service  # Keep for compatibility
            
            # Initialize chat history
            self.chat_history = ChatHistory()
            self.chat_history.add_system_message(
                "You are a helpful AI assistant with access to uploaded documents. "
                "Use the provided context from documents when relevant to answer user questions. "
                "Always be clear about whether you're using information from the documents or your general knowledge."
            )
            
            st.session_state.app_initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize services: {str(e)}")
            st.stop()
    
    async def get_embedding(self, text: str):
        """Generate embedding using direct Azure OpenAI client"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    async def test_embedding_service(self):
        """Test embedding service with simple text"""
        try:
            test_text = "This is a simple test sentence."
            st.write(f"Testing embedding service with: '{test_text}'")
            st.write(f"Text type: {type(test_text)}")
            st.write(f"Text length: {len(test_text)}")
            st.write(f"Text repr: {repr(test_text)}")
            
            # Check if text is properly encoded
            try:
                encoded = test_text.encode('utf-8')
                st.write(f"UTF-8 encoding works: {len(encoded)} bytes")
            except Exception as enc_error:
                st.error(f"UTF-8 encoding failed: {enc_error}")
                return False
            
            # Try to generate embedding directly with our custom function
            st.write("Calling direct embedding function...")
            embedding_result = await self.get_embedding(test_text)
            st.success(f"✅ Direct embedding service works! Generated {len(embedding_result)} dimensions")
            return True
            
        except Exception as e:
            st.error(f"❌ Embedding service test failed: {str(e)}")
            st.write(f"Error type: {type(e)}")
            
            # Try with a different approach - use the OpenAI client directly
            try:
                st.write("Trying direct Azure OpenAI client...")
                import openai
                from openai import AzureOpenAI
                
                # Use the endpoint as-is for direct Azure OpenAI client
                clean_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                st.write(f"Using endpoint: {clean_endpoint}")
                
                client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2024-06-01",
                    azure_endpoint=clean_endpoint
                )
                
                response = client.embeddings.create(
                    input="This is a test sentence.",
                    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
                )
                
                st.success(f"✅ Direct Azure OpenAI client works! Generated {len(response.data[0].embedding)} dimensions")
                st.error("Issue is with Semantic Kernel's AzureTextEmbedding wrapper")
                return False
                
            except Exception as direct_error:
                st.error(f"❌ Direct Azure OpenAI client also failed: {direct_error}")
                st.write("**Environment Variables Check:**")
                st.write(f"- AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET')}")
                st.write(f"- AZURE_OPENAI_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'NOT SET')}")
                st.write(f"- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'NOT SET')}")
                st.write(f"- API Key set: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
                
                st.error("**Solutions:**")
                st.write("1. **Check your Azure OpenAI resource** for deployed models")
                st.write("2. **Deploy an embedding model** (e.g., text-embedding-ada-002 or text-embedding-3-large)")
                st.write("3. **Update your .env file** with the correct deployment name")
                st.write("4. Common embedding deployment names: text-embedding-ada-002, embedding")
                
                # Try to list available deployments
                try:
                    st.write("**Trying to list available deployments...**")
                    deployments_response = client.models.list()
                    available_models = [model.id for model in deployments_response.data]
                    st.write(f"Available models: {available_models}")
                except Exception as list_error:
                    st.write(f"Could not list deployments: {list_error}")
                
                return False
    
    async def process_document(self, uploaded_file) -> bool:
        """Process uploaded document with comprehensive error handling"""
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                # Extract text
                text = self.document_processor.extract_text(tmp_path)
                
                if not text.strip():
                    st.warning(f"No text content found in {uploaded_file.name}")
                    return False
                
                # Chunk text
                chunks = self.document_processor.chunk_text(text, chunk_size=800, overlap=100)
                
                if not chunks:
                    st.warning(f"Could not create chunks from {uploaded_file.name}")
                    return False
                
                # Store chunks with embedding (try semantic memory first, fallback to simple storage)
                progress_bar = st.progress(0)
                successful_chunks = 0
                use_embeddings = True
                
                for i, chunk in enumerate(chunks):
                    try:
                        # Validate chunk
                        if not chunk or len(chunk.strip()) < 5:
                            continue
                        
                        # Clean chunk for embedding
                        clean_chunk = self.document_processor.clean_text(chunk)
                        if not clean_chunk:
                            continue
                            
                        doc_id = f"{uploaded_file.name}_chunk_{i}"
                        
                        if use_embeddings:
                            try:
                                # Generate embedding using our direct method
                                embedding = await self.get_embedding(clean_chunk)
                                # Store in document store with embedding
                                self.document_store[doc_id] = {
                                    'text': clean_chunk,
                                    'filename': uploaded_file.name,
                                    'chunk_id': i,
                                    'embedding': embedding
                                }
                                successful_chunks += 1
                            except Exception as embed_error:
                                st.warning(f"Embedding failed for chunk {i+1}, switching to simple storage")
                                use_embeddings = False  # Switch to fallback for remaining chunks
                                # Store in simple storage instead
                                self.document_store[doc_id] = {
                                    'text': clean_chunk,
                                    'filename': uploaded_file.name,
                                    'chunk_id': i
                                }
                                successful_chunks += 1
                        else:
                            # Use simple storage
                            self.document_store[doc_id] = {
                                'text': clean_chunk,
                                'filename': uploaded_file.name,
                                'chunk_id': i
                            }
                            successful_chunks += 1
                            
                    except Exception as e:
                        st.warning(f"Skipped chunk {i+1}: {str(e)[:100]}")
                        continue
                    progress_bar.progress((i + 1) / len(chunks))
                
                if successful_chunks == 0:
                    st.error(f"Failed to process any chunks from {uploaded_file.name}")
                    return False
                
                # Store which method was used
                st.session_state.using_embeddings = use_embeddings
                if use_embeddings:
                    st.success("✅ Using semantic search with embeddings")
                else:
                    st.info("ℹ️ Using simple keyword search (embeddings failed)")
                
                # Update session state
                st.session_state.processed_documents[uploaded_file.name] = {
                    'chunks': successful_chunks,
                    'size': len(text),
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.document_chunks_count += successful_chunks
                
                return True
                
            finally:
                # Cleanup temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return False
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        return dot_product / (magnitude1 * magnitude2)
    
    async def get_relevant_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context using embeddings if available, otherwise keyword search"""
        if st.session_state.document_chunks_count == 0:
            return ""
            
        try:
            # Try semantic search with our direct embeddings
            if getattr(st.session_state, 'using_embeddings', False):
                try:
                    # Generate embedding for query
                    query_embedding = await self.get_embedding(query)
                    
                    # Calculate similarity scores
                    scored_chunks = []
                    for doc_id, doc_data in self.document_store.items():
                        if 'embedding' in doc_data:
                            similarity = self.cosine_similarity(query_embedding, doc_data['embedding'])
                            if similarity > 0.7:  # Min relevance threshold
                                scored_chunks.append((similarity, doc_data['text']))
                    
                    # Sort by similarity and take top results
                    scored_chunks.sort(reverse=True, key=lambda x: x[0])
                    top_chunks = scored_chunks[:max_results]
                    
                    if top_chunks:
                        context_parts = []
                        for similarity, text in top_chunks:
                            context_parts.append(f"[Similarity: {similarity:.2f}] {text}")
                        return "\n\n---\n\n".join(context_parts)
                        
                except Exception as e:
                    st.warning(f"Semantic search failed, falling back to keyword search: {str(e)[:100]}")
            
            # Fallback to keyword-based search
            if not self.document_store:
                return ""
                
            query_lower = query.lower()
            scored_chunks = []
            
            for doc_id, doc_data in self.document_store.items():
                text = doc_data['text'].lower()
                score = sum(word in text for word in query_lower.split())
                if score > 0:
                    scored_chunks.append((score, doc_data['text']))
            
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            top_chunks = scored_chunks[:max_results]
            
            if not top_chunks:
                return ""
                
            context_parts = []
            for score, text in top_chunks:
                context_parts.append(f"[Matches: {score}] {text}")
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return ""
    
    async def chat_with_context(self, user_message: str, settings: Dict) -> str:
        """Enhanced chat with document context integration"""
        try:
            # Get relevant context from documents
            context = await self.get_relevant_context(user_message)
            
            # Create enhanced prompt template
            if context:
                prompt_template = """You are a helpful AI assistant with access to document content.

Use the following context from uploaded documents to help answer the user's question when relevant:

DOCUMENT CONTEXT:
{{$context}}

CHAT HISTORY:
{{$history}}

USER QUESTION: {{$user_input}}

Provide a helpful response. If you use information from the documents, mention that you're referencing the uploaded content. If the documents don't contain relevant information, use your general knowledge and make that clear."""
            else:
                prompt_template = """You are a helpful AI assistant.

CHAT HISTORY:
{{$history}}

USER QUESTION: {{$user_input}}

Provide a helpful response using your general knowledge."""
            
            # Configure prompt
            input_variables = [
                InputVariable(name="history", description="Chat history", is_required=True),
                InputVariable(name="user_input", description="User message", is_required=True),
            ]
            
            if context:
                input_variables.append(
                    InputVariable(name="context", description="Document context", is_required=True)
                )
            
            execution_settings = AzureChatPromptExecutionSettings(
                service_id="chat",
                max_tokens=settings.get('max_tokens', 1500),
                temperature=settings.get('temperature', 0.7)
            )
            
            prompt_config = PromptTemplateConfig(
                template=prompt_template,
                template_format="semantic-kernel",
                input_variables=input_variables,
                execution_settings=execution_settings
            )
            
            # Create function from prompt template
            from semantic_kernel.functions import KernelFunction
            chat_function = KernelFunction.from_prompt(
                prompt=prompt_template,
                plugin_name="ChatPlugin",
                function_name="chat_function",
                description="Chat with document context"
            )
            
            # Update chat history
            self.chat_history.add_user_message(user_message)
            
            # Create arguments
            arguments = KernelArguments(
                user_input=user_message,
                history=self.chat_history
            )
            
            if context:
                arguments["context"] = context
            
            # Get response
            result = await self.kernel.invoke(chat_function, arguments)
            response = str(result)
            
            # Update chat history
            self.chat_history.add_assistant_message(response)
            
            # Update metrics
            self.update_metrics(user_message, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def update_metrics(self, user_message: str, response: str):
        """Update token and cost metrics"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            user_tokens = len(encoding.encode(user_message))
            response_tokens = len(encoding.encode(response))
            total_tokens = user_tokens + response_tokens
            
            st.session_state.total_tokens += total_tokens
            
            # Rough cost estimation (adjust based on your Azure pricing)
            cost_per_1k_tokens = 0.002  # Example rate
            st.session_state.estimated_cost += (total_tokens / 1000) * cost_per_1k_tokens
            
        except Exception:
            pass  # Don't fail on metrics


def render_sidebar():
    """Render sidebar with controls and metrics"""
    with st.sidebar:
        st.header("📊 Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
            st.metric("Documents", len(st.session_state.processed_documents))
        with col2:
            st.metric("Cost (Est.)", f"${st.session_state.estimated_cost:.4f}")
            st.metric("Doc Chunks", st.session_state.document_chunks_count)
        
        st.header("⚙️ Settings")
        settings = {
            'temperature': st.slider("Temperature", 0.0, 2.0, 0.7, 0.1),
            'max_tokens': st.slider("Max Tokens", 100, 4000, 1500, 100)
        }
        
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'docx', 'csv', 'xlsx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_documents:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if asyncio.run(st.session_state.app.process_document(uploaded_file)):
                            st.success(f"✅ Processed {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                            
        # Show processed documents
        if st.session_state.processed_documents:
            st.header("📋 Processed Documents")
            for doc_name, info in st.session_state.processed_documents.items():
                with st.expander(f"📄 {doc_name}"):
                    st.write(f"**Chunks:** {info['chunks']}")
                    st.write(f"**Size:** {info['size']:,} chars")
                    st.write(f"**Processed:** {info['processed_at']}")
        
        # Show current environment variables with debugging
        st.write("**Current Environment Debug:**")
        st.write(f"Embedding deployment from os.getenv: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
        
        # Force reload .env and check
        from pathlib import Path
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / '.env'
        st.write(f"Loading .env from: {env_path}")
        load_dotenv(env_path, override=True)  # Force override
        st.write(f"After reload: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
        
        # Show .env file content
        if env_path.exists():
            with open(env_path) as f:
                env_content = f.read()
                st.text_area("Current .env file content:", env_content, height=100)
        
        # Test embedding service button
        if st.button("🧪 Test Embedding Service"):
            if asyncio.run(st.session_state.app.test_embedding_service()):
                st.info("Embedding service is working. You can enable semantic search.")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            for key in ['chat_messages', 'processed_documents', 'total_tokens', 'estimated_cost', 'document_chunks_count']:
                st.session_state[key] = [] if key == 'chat_messages' else ({} if key == 'processed_documents' else 0)
            st.rerun()
            
        return settings


def main():
    st.set_page_config(
        page_title="Enhanced AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Enhanced AI Assistant with RAG")
    st.caption("Upload documents and chat with AI that can reference your content")
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = EnhancedAIApp()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents or general questions..."):
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(
                        st.session_state.app.chat_with_context(prompt, settings)
                    )
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_response = f"I encountered an error: {str(e)}"
                    st.error(error_response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_response})


if __name__ == "__main__":
    main()