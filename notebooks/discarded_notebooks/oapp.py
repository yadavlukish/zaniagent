import streamlit as st
import PyPDF2
import openai
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_text = ""
    reader = PyPDF2.PdfFileReader(file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        pdf_text += page.extractText()
    return pdf_text

# Function to query OpenAI API
def query_openai(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Function to post a message to Slack
def post_to_slack(message, slack_token, channel):
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        st.error(f"Error posting to Slack: {e.response['error']}")
        return None

# Function to perform RAG (Retrieval-Augmented Generation)
def rag_pipeline(document, question, api_key):
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
    answer = query_openai(prompt, api_key)
    return answer

# Streamlit app
st.title("PDF Q&A with OpenAI and Slack")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# OpenAI API key input
api_key = st.text_input("OpenAI API Key", type="password")

# Slack token input
slack_token = st.text_input("Slack Token", type="password")

# Slack channel input
slack_channel = st.text_input("Slack Channel")

# Question input
question = st.text_input("Question")

# Submit button
if st.button("Submit"):
    if pdf_file and api_key and slack_token and slack_channel and question:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)

        # Use RAG pipeline to get the answer
        answer = rag_pipeline(pdf_text, question, api_key)

        # Post answer to Slack
        response = post_to_slack(answer, slack_token, slack_channel)
        if response:
            st.success("Answer posted to Slack!")
        else:
            st.error("Failed to post answer to Slack.")
    else:
        st.error("Please provide all required inputs.")
