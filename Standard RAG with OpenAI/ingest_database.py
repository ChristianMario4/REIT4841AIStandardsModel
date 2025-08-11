import os
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone.models import ServerlessSpec
import streamlit as st


# Initialise API_Keys for Pinecone DB and Google Gemini AI
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Check for existence of key, and if not, error in the chatbot UI
if not PINECONE_API_KEY:
    st.error("ERROR: PINECONE_API_KEY not functional, check Streamlit secrets")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("ERROR: GOOGLE_API_KEY not functional, check Streamlit secrets")
    st.stop()

# Define DATA_PATH and do checks, assisted use of os developed with Claude
DATA_PATH = r"data"

if not os.path.exists(DATA_PATH):
    st.error(f"Data directory '{DATA_PATH}' not found. Please check repo structure and contents.")
    st.info("Stopping chatbot")
    st.stop()

# Initialise Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialise the index for Pinecone DB -> code from Pinecone Docos
# http://app.pinecone.io/organizations/-OWEHQS7XzaN5h_xkrqU/projects/96720a04-95b1-489f-b947-fab8f69ee540/create-index/serverless
index_name = "reit4841-aistandards"

if not pc.has_index(index_name):
    st.info(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    st.success("New index created")
    
st.info(f"Using existing index: {index_name}")
index = pc.Index(index_name)

# Initialise GoogleGeminiAI embeddings
# Specify text_key as text to allow for referral to original text being used in response later
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                          google_api_key=GOOGLE_API_KEY,
                                          text_key = "text")

# Initialise Vector Store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialise document organiser
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, ids=uuids)
