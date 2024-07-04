import json
import os
from PyPDF2 import PdfReader
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
slack_token = os.getenv('SLACK_TOKEN')
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')

def extract_text_from_pdf(file):
    pdfreader = PdfReader(file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def create_vector_retriever(raw_text, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def create_llm_agent(llm_name):
    if llm_name == 'OpenAI':
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)
    else:
        return HuggingFaceEndpoint(
            repo_id=llm_name, max_length=128, temperature=0.5,token=HUGGINGFACEHUB_API_TOKEN
        )

def select_embeddings(llm_name):
    if llm_name == 'OpenAI':
        return OpenAIEmbeddings()
    else:
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {'allow_download': 'True'}
        return GPT4AllEmbeddings()

def create_chain(llm, retriever):
    template = """Answer the question based only on the following context:
    {context}
    Answers should be word to word match if the question is a word to word match
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | llm | output_parser

    return chain

def query_chain(chain, question):
    return chain.invoke(question)

def post_to_slack(message, channel):
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        return None
