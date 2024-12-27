import streamlit as st
from openai import OpenAI
from typing import Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIAssistant:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.getenv('Osk-KNkdu47PWzkH204si11boFen5T7jfQC5TD9_uKvLwRT3BlbkFJ8NaKSp6M0zJQNcG21wWv2J5ofyWvPyE2AuliYzFPQA')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Configure available models
        self.models = {
            "GPT-4 Turbo": "gpt-4-0125-preview",  # Latest GPT-4 Turbo model
            "GPT-4": "gpt-4",
            "GPT-3.5 Turbo": "gpt-3.5-turbo-0125"  # Latest GPT-3.5 Turbo model
        }

    def generate_response(self, 
                         prompt: str, 
                         model: str, 
                         enable_code_interpreter: bool, 
                         enable_file_search: bool,
                         max_tokens: int = 500) -> Optional[str]:
        """
        Generate response using OpenAI API
        """
        try:
            # Prepare system message based on enabled features
            system_message = "You are a helpful AI assistant."
            if enable_code_interpreter:
                system_message += " You can interpret and explain code."
            if enable_file_search:
                system_message += " You can assist with file-related queries."

            # Create chat completion with the latest API structure
            response = self.client.chat.completions.create(
                model=self.models[model],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "text"}  # New parameter for response format
            )
            
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = "GPT-4 Turbo"

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🤖 AI Assistant")
    
    try:
        # Sidebar configuration
        with st.sidebar:
            st.header("Settings")
            model = st.selectbox(
                "Select Model",
                ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo"],
                index=["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo"].index(st.session_state.model)
            )
            st.session_state.model = model
            
            max_tokens = st.slider("Max Response Length", 100, 4000, 500)
            enable_code_interpreter = st.checkbox("Enable Code Interpreter")
            enable_file_search = st.checkbox("Enable File Search")
            
            # Add model information
            st.divider()
            st.markdown("### Model Information")
            if model == "GPT-4 Turbo":
                st.markdown("- Latest and most capable model\n- Best for complex tasks\n- Larger context window")
            elif model == "GPT-4":
                st.markdown("- Highly capable model\n- Good for most tasks\n- Standard context window")
            else:
                st.markdown("- Fastest and most cost-effective\n- Good for simpler tasks\n- Standard context window")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

        # Initialize OpenAI Assistant
        assistant = OpenAIAssistant()
        
        # Chat interface
        for message in st.session_state.chat_history:
            role = "🧑‍💻 User" if message["role"] == "user" else "🤖 Assistant"
            with st.chat_message(message["role"]):
                st.write(f"{role}: {message['content']}")
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(f"🧑‍💻 User: {user_input}")
            
            # Generate and display response
            with st.spinner(f"Thinking using {model}..."):
                response = assistant.generate_response(
                    user_input,
                    model,
                    enable_code_interpreter,
                    enable_file_search,
                    max_tokens
                )
                
                if response:
                    with st.chat_message("assistant"):
                        st.write(f"🤖 Assistant: {response}")
                    
                    # Update chat history
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ])
    
    except Exception as e:
        st.error(f"An error occurred in the main application: {str(e)}")

if __name__ == "__main__":
    main()