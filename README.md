# Enhanced AI Assistant with RAG

A powerful AI chatbot built with Streamlit and Azure OpenAI that supports document upload and retrieval-augmented generation (RAG). Upload documents and chat with an AI that can reference your content to provide contextually relevant responses.

## Features

- **Multi-format document support**: PDF, DOCX, TXT, CSV, XLSX
- **Smart document processing**: Token-aware chunking with text cleaning
- **Dual search capabilities**: Semantic search with embedding fallback to keyword search
- **Azure OpenAI integration**: GPT-4 chat with comprehensive error handling
- **Interactive Streamlit UI**: Real-time metrics, debugging tools, and file management
- **Secure credential management**: Environment variables with .gitignore protection

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tuli-Keidar/general-purpose-chatbot.git
   cd general-purpose-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit python-dotenv semantic-kernel PyPDF2 python-docx pandas tiktoken numpy openai
   ```

3. **Configure credentials**:
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```

4. **Run the application**:
   ```bash
   streamlit run enhanced_ai_app.py
   ```

## Configuration

Edit `.env` with your Azure OpenAI details:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=your-chat-model-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-model-deployment
```

## File Structure

```
├── enhanced_ai_app.py      # Main Streamlit application
├── .env.example           # Template for environment variables
├── .gitignore            # Git ignore rules (protects .env)
└── README.md             # This file
```

## Usage

1. **Start the app** and configure your settings in the sidebar
2. **Upload documents** using the file uploader (supports multiple files)
3. **Chat with the AI** - it will use document content when relevant
4. **Monitor metrics** - track tokens, costs, and document processing
5. **Debug if needed** - built-in embedding service testing

## Technical Details

- **Document Processing**: Uses tiktoken for token-aware chunking
- **Search**: Dual-mode with semantic embeddings (cosine similarity) and keyword fallback  
- **Error Handling**: Comprehensive fallback mechanisms for robust operation
- **Security**: Credentials stored in environment variables, excluded from git

## Dependencies

- streamlit
- semantic-kernel
- azure-openai
- PyPDF2, python-docx, pandas (document processing)
- tiktoken, numpy (text processing and embeddings)
