import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
from snowflake.snowpark import Session
import pandas as pd
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()




# # Snowflake Cortex configurations from .env
# SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
# SNOWFLAKE_USER_PASSWORD = os.getenv("SNOWFLAKE_USER_PASSWORD")
# SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
# SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
# SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
# SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
# SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
# CORTEX_SEARCH_SERVICE = os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")

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

if "snowflake_session" not in st.session_state:
    st.session_state.snowflake_session = Session.builder.configs(connection_params).create()

session = st.session_state.snowflake_session

# session = Session.builder.configs(connection_params).create()

# Set pandas display options
pd.set_option("max_colwidth", None)

### Default Values
NUM_CHUNKS = 9  # Number of chunks provided as context
SLIDE_WINDOW = 7  # Number of last conversations to remember

# Red flag keywords
RED_FLAGS = ["suicide", "harm", "hopeless", "kill myself", "worthless", "cut myself"]

# Columns to query in the service
COLUMNS = ["chunk", "relative_path", "category"]

# # Initialize Snowflake session and Cortex search service
# session = Session.builder.configs(connection_parameters).create()
# session = get_active_session()
root = Root(session)
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]


### Functions

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

def reset_session():
    """Reset the chatbot session."""
    st.session_state.clear()
    st.session_state.messages = []

def init_session():
    """Initialize session variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def analyze_red_flags(user_input):
    """Check if the user's input contains dangerous keywords and respond appropriately."""
    for flag in RED_FLAGS:
        if flag in user_input.lower():
            return (
                True,
                "I'm really sorry you're feeling this way. You're not alone, and help is available. "
                "Please consider reaching out to a trusted individual or a helpline. The sidebar has a list of Mental Health Helplines."
                "Would you like me to provide emergency resources?"
            )
    return False, ""

def get_similar_chunks_search_service(query):
    """Fetch context-related chunks from the search service."""
    response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
    return response.json()

def get_chat_history():
    """Retrieve a summarized chat history."""
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    return st.session_state.messages[start_index:]

def summarize_chat_history(chat_history, question):
    """Generate a natural-language summary of the chat history and question."""
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
    summary = Complete("mistral-large2", prompt)
    return summary.strip()

def create_prompt(user_input):
    """Create the chatbot's response prompt."""
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in get_chat_history()])
    if chat_history:
        summarized_input = summarize_chat_history(chat_history, user_input)
        prompt_context = get_similar_chunks_search_service(summarized_input)
    else:
        prompt_context = get_similar_chunks_search_service(user_input)

    prompt = f"""
        You are a compassionate and experienced psychiatrist who provides tailored and empathetic responses. 
        Use the CONTEXT below to assist the user. Speak naturally, compassionately, and with sensitivity. Please don't assume anything about relations. Only refer to chat history if there are any relations.

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

def generate_response(user_input):
    """Generate the chatbot's response to the user's input."""
    # Check for red flags
    red_flag_detected, red_flag_response = analyze_red_flags(user_input)
    if red_flag_detected:
        return red_flag_response

    # Otherwise, proceed with the main response generation
    prompt, relative_paths = create_prompt(user_input)
    response = Complete("mistral-large2", prompt)
    return response.strip()

### Main App

def main():
    st.title(":speech_balloon: Emotional Health Support Chatbot")
    st.markdown(
        "Welcome to the Emotional Health Support Chatbot. This is a safe space to share your feelings and thoughts. ❤️"
    )

    # Sidebar for configuration options
    config_options()

    # Initialize session
    init_session()

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("How can I support you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display chatbot's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("..."):
                response = generate_response(user_input)
                message_placeholder.markdown(response)

        # Add chatbot's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
