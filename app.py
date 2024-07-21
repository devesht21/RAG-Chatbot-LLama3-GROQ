import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv

load_dotenv()
## load the GROQ API Key
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)


def create_vector_embedding(url):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        st.session_state.loader = WebBaseLoader(url)  ## Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )


st.title("RAG Article Q&A With Groq And Llama3")

st.sidebar.title("URL")

url = st.sidebar.text_input("Enter the Url")

if st.sidebar.button("Load the article"):
    create_vector_embedding(url=url)
    st.sidebar.write("Article is loaded")

user_prompt = st.text_input("Enter your query from the Article")


import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response["answer"])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")
