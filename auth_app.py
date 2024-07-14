
from imports import *

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


with open('credentials.yaml') as credentials_file:
    credentials = yaml.safe_load(credentials_file)

authenticator = stauth.Authenticate(
    credentials['credentials'],
    credentials['cookie']['name'],
    credentials['cookie']['key'],
    credentials['cookie']['expiry_days'],
    credentials['preauthorized']
)   

#name, authentication_status, username = authenticator.login('Login', 'main')
# Use 'fields' instead of 'form_name'
name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'login'})
# Use 'fields' instead of 'form_name'
#name, authentication_status, username = authenticator.login('Login')

# Function to manage question limit
def can_ask_question(username):
        current_time = time.time()
        timestamps = st.session_state.question_count[username]['timestamps']
        # Filter out timestamps older than 1 hour
        timestamps = [ts for ts in timestamps if current_time - ts < 3600]
        st.session_state.question_count[username]['timestamps'] = timestamps

        if len(timestamps) < 1:
            st.session_state.question_count[username]['timestamps'].append(current_time)
            return True
        else:
            return False



if authentication_status:
    # Initialize session state for question count and timestamps
    if 'question_count' not in st.session_state:
        st.session_state.question_count = {}
    
    if username not in st.session_state.question_count:
        st.session_state.question_count[username] = {'count': 0, 'timestamps': []}



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
            
        if can_ask_question(username):
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

        else:
            st.error("Question limit exceeded. Please try again later.")

    authenticator.logout('Logout', 'sidebar')   

elif authentication_status == False:
    st.error('Username or password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')        



