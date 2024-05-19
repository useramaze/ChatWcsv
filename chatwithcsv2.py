import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import base64

# Define your API key here
headers ={
    "authorization": st.secrets["API_KEY"],
    "content-type": "application/json"
}

os.environ["PANDASAI_API_KEY"] = headers["authorization"]

DEFAULT_CSV_PATH = "Default_Accident_Data.csv"


# Dictionary to store the extracted dataframes
data = {}

def main():
    st.set_page_config(page_title="DataAssistant", page_icon="🐼")
    st.title("Chat with Your Data ")
    
    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Please ensure the first row has the column names.*]")

    # Load the default CSV file if no file is uploaded
    if file_upload is None:
        st.info(f"Upload your own file from sidebar for Customized Reports")
        data = extract_dataframes(DEFAULT_CSV_PATH)
    else:
        data = extract_dataframes(file_upload)

    df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
    st.dataframe(data[df])

    # Instantiate the BambooLLM
    llm = BambooLLM()
    
    # Instantiate the PandasAI agent
    analyst = get_agent(data, llm)

    # Start the chat with the PandasAI agent
    chat_window(analyst, data[df])

def chat_window(analyst, df):
    with st.chat_message("assistant"):
        st.text("Get instant answers to your runtime queries with Data Assistant.")

    # Initializing message history and chart path in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message["response"])
            elif 'error' in message:
                st.text(message['error'])
            elif 'plot_data' in message:
                img = base64.b64decode(message['plot_data'])
                st.image(img)



    # Predefined questions for the user to click
    predefined_questions = [ 
        "Can you plot the number of accidents over the years?",
        "What are the top 5 districts suffering from Road Accidents?",
        "What are the top 3 Accident Sublocations for Road Accidents?",
        "What are the top 3 Collision Type causing Fatal Severity Road Accidents?"
    ]

    st.markdown("## Sample Questions:")

    # Display the predefined questions as buttons
    for question in predefined_questions:
        if st.button(question):
            process_question(analyst, question)

    # Explicit user queries
    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        process_question(analyst, user_question)

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    # Button to clear history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def process_question(analyst, question):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "question": question})

    try:
        with st.spinner("Analyzing..."):
            response = analyst.chat(question)

            # Check if a plot has been generated and saved in the export directory
            plot_path = "exports/charts/temp_chart.png"
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                st.image(base64.b64decode(img_data))
                st.session_state.messages.append({"role": "assistant", "plot_data": img_data})
                os.remove(plot_path)
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "response": response})

    except Exception as e:
        st.write(e)
        error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
        st.session_state.messages.append({"role": "assistant", "error": error_message})

def get_agent(data, llm):
    """
    The function creates an agent on the dataframes extracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm: LLM object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return agent

def extract_dataframes(file_path_or_buffer):
    """
    This function extracts dataframes from the given file path or buffer.
    Args: 
        file_path_or_buffer: Path to the file or file buffer (uploaded file)
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs: A dictionary with the dataframes
    """
    dfs = {}
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.endswith('.csv'):
            df = pd.read_csv(file_path_or_buffer)
            dfs["Default Data"] = df
        elif file_path_or_buffer.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(file_path_or_buffer)
            for sheet_name in xls.sheet_names:
                dfs[sheet_name] = pd.read_excel(file_path_or_buffer, sheet_name=sheet_name)
    else:
        if file_path_or_buffer.name.endswith('.csv'):
            csv_name = file_path_or_buffer.name.split('.')[0]
            df = pd.read_csv(file_path_or_buffer)
            dfs[csv_name] = df
        elif file_path_or_buffer.name.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(file_path_or_buffer)
            for sheet_name in xls.sheet_names:
                dfs[sheet_name] = pd.read_excel(file_path_or_buffer, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()
