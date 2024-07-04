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
from openai.error import RateLimitError

# Load environment variables from .env file
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
slack_token = os.getenv('SLACK_TOKEN')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_text = ""
    reader = PyPDF2.PdfReader(file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        pdf_text += page.extract_text()
    return pdf_text

# Function to query OpenAI API using the new API format with retry logic
def query_openai(prompt):
    openai.api_key = openai_api_key
    retries = 5
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message['content'].strip()
        except RateLimitError:
            if i < retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                st.error("API rate limit exceeded. Please try again later.")
                return None

# Function to post a message to Slack
def post_to_slack(message, channel):
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        st.error(f"Error posting to Slack: {e.response['error']}")
        return None

# Function to perform RAG (Retrieval-Augmented Generation)
def rag_pipeline(document, question):
    # Split document into chunks (e.g., paragraphs)
    chunks = document.split('\n\n')
    
    # Retrieve relevant chunks based on the question
    relevant_chunks = []
    for chunk in chunks:
        if question.lower() in chunk.lower():
            relevant_chunks.append(chunk)
    
    # Combine relevant chunks
    combined_text = "\n\n".join(relevant_chunks)
    
    # Generate answer using OpenAI
    prompt = f"Document: {combined_text}\n\nQuestion: {question}\n\nAnswer:"
    answer = query_openai(prompt)
    return answer

# Streamlit app
st.title("PDF Q&A with OpenAI and Slack")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Slack channel input
slack_channel = st.text_input("Slack Channel")

# Question input
question = st.text_input("Question")

# Submit button
if st.button("Submit"):
    if pdf_file and slack_channel and question:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)

        # Use RAG pipeline to get the answer
        answer = rag_pipeline(pdf_text, question)

        # Post answer to Slack
        if answer:
            response = post_to_slack(answer, slack_channel)
            if response:
                st.success("Answer posted to Slack!")
            else:
                st.error("Failed to post answer to Slack.")
    else:
        st.error("Please provide all required inputs.")
