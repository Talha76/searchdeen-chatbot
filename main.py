import streamlit as st
from streamlit_chat import message
from llm import get_response

st.set_page_config(page_title="Chat App", page_icon="💬")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("💬 Streaming Chat App")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["is_user"], key=f"msg_{i}")

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    message(user_input, is_user=True, seed=42, key=f"msg_{len(st.session_state.messages)}")
    
    # Add user message to history first
    st.session_state.messages.append({"content": user_input, "is_user": True})
    
    # Prepare history for the LLM
    history = [
        (
            "user" if msg["is_user"] else "ai",
            msg["content"]
        )
        for msg in st.session_state.messages[:-1]  # Exclude the current user message
    ]
    
    # Create a placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in get_response(user_input, history):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add bot response to history
    st.session_state.messages.append({"content": full_response, "is_user": False})
    
    # Rerun to display with streamlit_chat components
    st.rerun()
