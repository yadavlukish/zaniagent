import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from slack_sdk import WebClient
import json

import streamlit as st
import PyPDF2
import openai
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import os
import time
import openai
#from openai.error import RateLimitError

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

from langchain.retrievers import BM25Retriever, EnsembleRetriever

from getpass import getpass
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
slack_token = os.getenv('SLACK_TOKEN')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    # provide the path of  pdf file/files.
    pdfreader = PdfReader('handbook.pdf')
    from typing_extensions import Concatenate
        # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

# Function to create vector database
def create_vector_retriever(raw_text):
     # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = RecursiveCharacterTextSplitter(
    #separator = "\n",
    chunk_size = 250,
    chunk_overlap  = 50,
    length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def create_llm_agent():
    repo_id = "tiiuae/falcon-7b-instruct"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )
    return llm

# Function to query OpenAI API using the new API format with retry logic
def create_chain(llm, retriever):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    #st.write(question)
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | llm | output_parser

    return chain


# Function to query OpenAI API using the new API format with retry logic
def query_chain(chain, question):
    return chain.invoke(question)


# Function to post a message to Slack
def post_to_slack(message, channel):
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        st.error(f"Error posting to Slack: {e.response['error']}")
        return None


# Streamlit app
st.title("PDF Q&A with OpenAI and Slack")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Slack channel input
slack_channel = st.text_input("Slack Channel","agent_updates")

# Question input
question = st.text_input("Question")

model_embedding_option = st.selectbox(
    "Select model and embedding?",
    ("Huggingface_GPT4all", "OpenAI"))

st.write("You selected:", model_embedding_option)

# Submit button
if st.button("Submit"):
    if pdf_file and slack_channel and question:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)

        # Create vector database
        vector_retriever = create_vector_retriever(pdf_text)

        llm=create_llm_agent()

        chain=create_chain(llm,vector_retriever)
        
 
        answer=query_chain(chain, question)


        # Post answer to Slack
        if answer:
            response = post_to_slack(answer, slack_channel)
            if response:
                st.success("Answer posted to Slack!")
            else:
                st.error("Failed to post answer to Slack.")
    else:
        st.error("Please provide all required inputs.")

