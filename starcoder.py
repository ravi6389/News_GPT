import streamlit as st
import pandas as pd
from transformers import pipeline
from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline
# from langchain import LLMChain
from langchain.chains import LLMChain

import os
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


# ---------------------------
# Load Hugging Face model and tokenizer
# ---------------------------

# Define the model you want to use (e.g., 'bigcode/starcoder', 'meta-llama/Llama-2-7b-chat-hf')
# MODEL_NAME = "bigcode/starcoder"

# @st.cache_resource
# def load_model(model_name):
#     """
#     Load the language model and tokenizer from Hugging Face.
#     This function is cached to prevent reloading on every interaction.
#     """
#     text_generator = pipeline("text-generation", model=model_name, max_length=512, temperature=0.3, top_p=0.95)
#     return text_generator

# # Load the Hugging Face pipeline for code generation
# text_generator = load_model(MODEL_NAME)

# Create a LangChain LLM using the Hugging Face pipeline
os.environ['SSL_CERT_FILE'] = 'C:\\Users\\RSPRASAD\\AppData\\Local\\.certifi\\cacert.pem'
GROQ_API_KEY = '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'
llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
# llm = HuggingFacePipeline(pipeline=text_generator)

# ---------------------------
# Define the Prompt Template
# ---------------------------

prompt_template = """
You are given a dataset with the following columns: {columns}.
Write a Python script using pandas to answer the following question: "{question}".
The data is stored in a pandas DataFrame called 'df'.
Store the result in a variable called 'result'.Dont give any description but
just the python code for the question that I can use in python exec function.
"""

# Create the PromptTemplate object using LangChain
template = PromptTemplate(
    input_variables=["columns", "question"],
    template=prompt_template,
)

# Create the LLMChain to manage the model and prompt interaction
llm_chain = LLMChain(prompt=template, llm=llm)

# llm_chain = llm |template

# ---------------------------
# Streamlit App Interface
# ---------------------------

st.title("📊 NewsGPT for Hindu Newspaper")

st.markdown("""
Developed by Ravi Shankar Prasad. 
The dataset has following structure.
""")
df = pd.read_excel\
("C:\\Users\\RSPRASAD\\OneDrive - Danaher\\Learning\\UPSC_Crawler\\data\\Summary.xlsx")

st.write(df.head(3))

st.write('Currently it has news items for 2nd, 3rd and 4th September.\
         You can ask questions like - Give me summary of news of 2024/9/3 or 2024/9/2.\
         The LLM will write a python script dynamically as per the question and \
         will execute it to give the answers.')
# CSV file uploader

# Read the uploaded CSV file into a pandas DataFrame
# df = pd.read_excel\
# ("C:\\Users\\RSPRASAD\\OneDrive - Danaher\\Learning\\UPSC_Crawler\\data\\Summary.xlsx")

# Display the first few rows of the dataset
st.write("### 📈 Uploaded Dataset:")
# st.dataframe(df.head())

# Ask the user for a question about the data
question = st.text_input("❓ Ask a question about the data")

if question:
    with st.spinner("Generating Python code..."):
        try:
            # Run the LLMChain to generate the Python script based on the question and CSV columns
            python_script = llm_chain.invoke({
                "columns": ", ".join(df.columns),
                "question": question
            })

            # Display the generated Python code
            st.write("### 📝 Generated Python Code:")
            st.code(python_script, language='python')
            
            # Option to execute the generated Python code
            if st.button("▶️ Run Code"):
                try:
                    # Provide the DataFrame in the exec environment
                    exec_globals = {"df": df, "pd": pd}
                    # st.write(python_script)
                    st.write(python_script['text'].strip('`'))
                    python_script['text'] = python_script['text'].strip('`')
                    exec(python_script['text'], exec_globals)
                    # st.write(exec_globals)
                    # If a result variable is present, display it
                    if 'result' in exec_globals:
                        st.write("### 📊 Result:")
                        st.write(exec_globals['result'])
                    else:
                        st.warning("⚠️ The code did not produce a 'result' variable.")
                except Exception as e:
                    st.error(f"🚫 Error running the code: {e}")
        except Exception as e:
            st.error(f"🚫 Error generating the code: {e}")
