
from imports import *

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

model_embedding = config["model_embedding"]

# Streamlit app
st.title("ZaniAGENT")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Select model and embeddings
model_embedding_option = st.selectbox(
    "Select model and embedding?",
    tuple(model_embedding))

st.write("You selected:", model_embedding_option)

# Slack channel input
slack_channel = st.text_input("Slack Channel","agent_updates")

# Question input
#questions = st.text_input("Question")

questions=st.text_area("Question","""What is the name of the company? 
Who is the CEO of the company? 
What is their vacation policy? 
What is the termination policy?""")

# Initialize session state for pdf_text, vector_retriever, and model_embedding_option
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'vector_retriever' not in st.session_state:
    st.session_state.vector_retriever = None
if 'last_uploaded_pdf' not in st.session_state:
    st.session_state.last_uploaded_pdf = None
if 'last_embedding_option' not in st.session_state:
    st.session_state.last_embedding_option = None

# Check if a new PDF file is uploaded or if the model embedding option is changed
if (pdf_file is not None and pdf_file != st.session_state.last_uploaded_pdf) or (model_embedding_option != st.session_state.last_embedding_option):
    st.session_state.last_uploaded_pdf = pdf_file
    st.session_state.last_embedding_option = model_embedding_option
    if pdf_file:
        # Extract text from PDF
        st.session_state.pdf_text = extract_text_from_pdf(pdf_file)
        selected_embeddings = select_embeddings(model_embedding_option)
        # Create vector database
        st.session_state.vector_retriever = create_vector_retriever(st.session_state.pdf_text, selected_embeddings)

# Submit button
if st.button("Submit"):
    if not st.session_state.pdf_text:
        st.error("Please upload a PDF file.")
        st.stop()  # Stop the execution here if no PDF is provided and no previous text is available

    if st.session_state.pdf_text and slack_channel and questions:
        llm = create_llm_agent(model_embedding_option)
        chain = create_chain(llm, st.session_state.vector_retriever)

        questions = questions.strip().split('\n')
        print(questions)
        
        answers={}
        for question in questions:
            answer=query_chain(chain, question)
            if not answer:
                answers[question] = "Data Not Available"
            else:
                answers[question] = answer

        output = {question: answers[question] for question in questions}
        output=json.dumps(output, indent=2)


        # Post answer to Slack
        if answer:
            response = post_to_slack(output, slack_channel)
            if response:
                st.success("Answer posted to Slack!")
            else:
                st.error("Failed to post answer to Slack.")
    else:
        st.error("Please provide all required inputs.")

