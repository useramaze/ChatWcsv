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

def main():
    st.set_page_config(page_title="DataAssistant", page_icon="🐼")
    st.title("Chat with Your Data ")
    
    # Initialize the data dictionary
    data = {}

    # Load default CSV file
    default_csv = 'ChatWithCSV PANDASAI\Merged_Clean_CSV.csv'
    if os.path.exists(default_csv):
        default_df = pd.read_csv(default_csv)
        data['Default Data'] = default_df

    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Please ensure the first row has the column names.*]")

    if file_upload is not None:
        uploaded_data = extract_dataframes(file_upload)
        data.update(uploaded_data)
    
    if data:
        df_key = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df_key])
    else:
        st.warning("Please upload a dataset to proceed.")

    # Instantiate the BambooLLM
    llm = BambooLLM()
    
    # Instantiate the PandasAI agent
    analyst = get_agent(data, llm)

    # Start the chat with the PandasAI agent
    if data:
        chat_window(analyst, data[df_key])

def chat_window(analyst, df):
    with st.chat_message("assistant"):
        st.text("Get instant answers to your runtime queries with Data Assistant.")

    # List of default questions
    default_questions = [
        "What are the top 5 districts suffering from Road Accidents?",
        "Can you plot the number of accidents over the years?",
        "What are the top 5 types of collisions causing road accidents?",
        "Which Road type causes the highest number of Fatal Accidents?"
    ]

    st.markdown("## Sample Questions:")
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None

    for i, question in enumerate(default_questions):
        if st.button(question, key=f"default_question_{i}"):
            st.session_state.selected_question = question
            st.session_state.question_triggered = True

    if st.session_state.selected_question and st.session_state.question_triggered:
        user_question = st.session_state.selected_question
        st.session_state.messages.append({"role": "user", "question": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
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
        st.session_state.selected_question = None
        st.session_state.question_triggered = False

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

    # Getting the questions from the users
    user_question = st.chat_input("What are you curious about?")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)

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

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    # Button to clear history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

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

def extract_dataframes(raw_file):
    """
    This function extracts dataframes from the uploaded file/files
    Args: 
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs: A dictionary with the dataframes
    """
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif raw_file.name.split('.')[1] == 'xlsx' or raw_file.name.split('.')[1] == 'xls':
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()
