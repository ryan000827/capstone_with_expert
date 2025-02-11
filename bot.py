import streamlit as st
from utils import write_message
from agent import generate_response
import pandas as pd

# Page Config
st.set_page_config("Caroline", page_icon=":star2:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, My name is Caroline and I'm here to help you with any concerns you and your partner might have for the future!  For a start, could you kindly introduce yourself and your partner?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("Type Here"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)

# Download chat history
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

chat_history = pd.DataFrame(st.session_state.messages)
csv = convert_df(chat_history)

st.download_button(
    label="Download chat history as CSV",
    data=csv,
    file_name="chat_history_df.csv",
    mime="text/csv",
)
