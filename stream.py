import os
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate  # Needed for PromptTemplate
from langchain_community.document_loaders import PDFMinerLoader
import streamlit as st

# Set API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize the chat model
model = init_chat_model("gemma2-9b-it", model_provider="groq", api_key=GROQ_API_KEY)

# Load PDF document
loader = PDFMinerLoader(r'C:\Users\HP\Downloads\RAG\Statistics Notes.pdf')
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings_model)
retriever = vectorstore.as_retriever()

# Define prompt template
prompt = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are an assistant for question-answering tasks. "
                    "Use the following retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Answer:"
                )
            )
        )
    ]
)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)



question = st.text_input('Enter question to ask')
response = rag_chain.invoke(question)
st.write(response)