import streamlit as st
import uuid  # To generate unique session IDs
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from snowflake.core import Root
import pandas as pd
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Snowflake connection parameters
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
}

CORTEX_SEARCH_DATABASE = "CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"

# Default configurations
NUM_CHUNKS = 9
SLIDE_WINDOW = 7
RED_FLAGS = ["suicide", "harm", "hopeless", "kill myself", "worthless", "cut myself"]

### Helper Functions
def config_options():
    """Configure chatbot options in the sidebar."""
    st.sidebar.selectbox("Select your model:", ["mistral-large2"], key="model_name")
    with st.sidebar.expander("📞 Mental Health Helplines"):
        st.markdown(
            """
            - **USA**: National Suicide Prevention Lifeline - 988  
            - **UK**: Samaritans - 116 123  
            - **India**: AASRA - 91-9820466726  
            - **Canada**: Talk Suicide Canada - 1-833-456-4566  
            - **Australia**: Lifeline - 13 11 14  
            - **Global**: Befrienders Worldwide - [Visit Website](https://www.befrienders.org)
            """
        )

    if st.sidebar.button("Restart Session", key="restart_session"):
        reset_session()

def get_session_id():
    """Generate or retrieve a unique session ID for the user."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Generate unique ID
    session_id = st.session_state.session_id

    # Initialize message history if not already present
    if f"{session_id}_messages" not in st.session_state:
        st.session_state[f"{session_id}_messages"] = []

    return session_id


def init_session():
    """Initialize the Snowflake session for the application."""
    if "global_snowflake_session" not in st.session_state:
        st.session_state["global_snowflake_session"] = Session.builder.configs(connection_params).create()
    return st.session_state["global_snowflake_session"]

def reset_session():
    """Reset the chatbot session for the current user."""
    session_id = get_session_id()
    # Clear user-specific messages
    st.session_state[f"{session_id}_messages"] = []
    # Close and remove the global Snowflake session if it exists
    if "global_snowflake_session" in st.session_state:
        st.session_state["global_snowflake_session"].close()
        del st.session_state["global_snowflake_session"]

def analyze_red_flags(user_input):
    """Check for dangerous keywords in user input."""
    for flag in RED_FLAGS:
        if flag in user_input.lower():
            return (
                True,
                "I'm really sorry you're feeling this way. You're not alone, and help is available. "
                "Please consider reaching out to a trusted individual or a helpline. The sidebar has a list of Mental Health Helplines."
            )
    return False, ""

def get_similar_chunks_search_service(query, svc):
    """Fetch context-related chunks from the search service."""
    response = svc.search(query, ["chunk", "relative_path", "category"], limit=NUM_CHUNKS)
    return response.json()

def get_chat_history(session_id):
    """Retrieve a summarized chat history for the user."""
    messages = st.session_state.get(f"{session_id}_messages", [])
    start_index = max(0, len(messages) - SLIDE_WINDOW)
    return messages[start_index:]

def summarize_chat_history(chat_history, question):
    """Generate a summary query based on chat history and the question."""
    prompt = f"""
        Based on the chat history below and the question, create a query that includes the context:
        
        <chat_history>
        {chat_history}
        </chat_history>
        
        <question>
        {question}
        </question>
        
        Only return the summary query.
    """
    # Pass the Snowflake session explicitly
    summary = Complete("mistral-large2", prompt, session=st.session_state["global_snowflake_session"])
    return summary.strip()

def create_prompt(user_input, session_id, svc):
    """Create the chatbot's response prompt."""
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in get_chat_history(session_id)])
    if chat_history:
        summarized_input = summarize_chat_history(chat_history, user_input)
        prompt_context = get_similar_chunks_search_service(summarized_input, svc)
    else:
        prompt_context = get_similar_chunks_search_service(user_input, svc)

    prompt = f"""
        You are a compassionate and experienced psychiatrist who provides tailored and empathetic responses. 
        Use the CONTEXT below to assist the user. Speak naturally, compassionately, and with sensitivity.

        CHAT HISTORY:
        {chat_history}
        
        CONTEXT:
        {prompt_context}

        User's Input:
        {user_input}

        Your Response:
    """
    json_data = json.loads(prompt_context)
    relative_paths = set(item["relative_path"] for item in json_data["results"])
    return prompt, relative_paths

def generate_response(user_input, session_id, svc):
    """Generate the chatbot's response."""
    # Check for red flags
    red_flag_detected, red_flag_response = analyze_red_flags(user_input)
    if red_flag_detected:
        return red_flag_response

    # Generate main response
    prompt, relative_paths = create_prompt(user_input, session_id, svc)
    # Pass the Snowflake session explicitly
    response = Complete("mistral-large2", prompt, session=st.session_state["global_snowflake_session"])
    return response.strip()

### Main Application

def main():
    st.title(":speech_balloon: Emotional Health Support Chatbot")
    st.markdown("Welcome to the Emotional Health Support Chatbot. This is a safe space to share your feelings. ❤️")

    # Sidebar for configuration options
    config_options()

    # Initialize user-specific session and Snowflake connection
    session_id = get_session_id()
    snowflake_session = init_session()

    # Access Cortex search service
    root = Root(snowflake_session)
    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

    # Display previous chat messages
    for message in st.session_state[f"{session_id}_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("How can I support you today?"):
        # Add user message to chat history
        st.session_state[f"{session_id}_messages"].append({"role": "user", "content": user_input})

        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display chatbot's response
        with st.chat_message("assistant"):
            with st.spinner("..."):
                response = generate_response(user_input, session_id, svc)
                st.markdown(response)

        # Add chatbot's response to chat history
        st.session_state[f"{session_id}_messages"].append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()
