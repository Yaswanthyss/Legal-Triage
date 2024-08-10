# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text



# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     #embeddings = GoogleGenerativeAIEmbeddings(model="models/jina-finetuned-question-answering")
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     #embeddings = GoogleGenerativeAIEmbeddings(model="models/jina-finetuned-question-answering")

#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     # st.write("Reply: ", response["output_text"])

#     st.write("Reply:")
#     with st.expander("See Reply"):
#         st.write(response["output_text"])

    




# def main():
#     st.set_page_config("Equity Research")
#     st.header("Equity Research Tool💁")

#     user_question = st.text_input("Do you wanna know something ?,Ask here")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
#     provided context just say, "Answer is not available in the context", don't provide the wrong answer.
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# def main():
#     st.set_page_config(page_title="Equity Research Tool", page_icon="💁", layout="wide")
    
#     st.sidebar.title("Menu")
#     pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    
#     if pdf_docs and st.sidebar.button("Submit & Process"):
#         with st.spinner("Processing..."):
#             raw_text = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#             st.sidebar.success("Processing complete!")
    
#     st.header("Equity Research Tool 💁")
#     st.write("This tool allows you to upload PDF files, process their content, and ask questions about the content.")

#     if 'response' not in st.session_state:
#         st.session_state.response = ""
    
#     user_question = st.text_input("Enter your question here:")
    
#     if user_question:
#         with st.spinner("Searching for the answer..."):
#             response = user_input(user_question)
#             st.session_state.response = response
    
#     if st.session_state.response:
#         st.write("### Question")
#         st.write(user_question)
#         st.write("### Reply")
#         st.write(st.session_state.response)

#     with st.sidebar.expander("Instructions"):
#         st.write("""
#             1. Upload one or more PDF files.
#             2. Click 'Submit & Process' to analyze the documents.
#             3. Enter a question in the text box and press Enter.
#             4. View the answer generated from the processed documents.
#         """)

# if __name__ == "__main__":
#     main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer.
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main function to build the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Equity Research Tool", page_icon="💁", layout="wide")

    # Sidebar menu
    st.sidebar.title("Menu")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)

    # Process PDFs
    if pdf_docs and st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("Processing complete!")

    # Main content
    st.title("Equity Research Tool 💁")
    st.markdown("---")
    st.write("This tool allows you to upload PDF files, process their content, and ask questions about the content.")

    # User input section
    user_question = st.text_input("Enter your question here:")
    if user_question:
        with st.spinner("Searching for the answer..."):
            response = user_input(user_question)
            st.success("Answer found!")
            st.markdown("---")
            st.write("### Your Question:")
            st.write(user_question)
            st.write("### Reply:")
            st.write(response)

    # Styling and additional UI elements
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions:")
    st.sidebar.markdown("""
        1. **Upload PDF Files:** Upload one or more PDF files using the sidebar.
        2. **Submit & Process:** Click the button to process the uploaded PDFs.
        3. **Enter Question:** Enter your question in the text box above and press Enter.
        4. **View Answer:** The answer will be displayed below your question.
    """)

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.write("Built with ❤️ by Yaswanth & Ponsankar")

if __name__ == "__main__":
    main()

