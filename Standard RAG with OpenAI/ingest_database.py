import os
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone.models import ServerlessSpec
import streamlit as st

# Get environment variables
GOOGLE_API_KEY = "AIzaSyCu3tkstKi4AkznUxuHGfTmmucaCqkuhho"
PINECONE_API_KEY = "pcsk_32nPYb_QEpPzLyBdoisRvfy5zmTP1ePgvJZ1qHvozYCSx8vUF7uMxAbhVN18tTwDDMmAyL"


# Check for existence of keys
if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found. Set environment variable or update script.")
    exit(1)
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found. Set environment variable or update script.")
    exit(1)

# Define DATA_PATH and do checks, assisted use of os developed with Claude
DATA_PATH = r"data"

if not os.path.exists(DATA_PATH):
    st.error(f"Data directory '{DATA_PATH}' not found. Please check repo structure and contents.")
    st.info("Stopping chatbot")
    st.stop()

def main(): 
    print("Starting document ingestion into Pinecone index")
    
    # Initialise Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Initialise the index for Pinecone DB -> code from Pinecone Docos
    # http://app.pinecone.io/organizations/-OWEHQS7XzaN5h_xkrqU/projects/96720a04-95b1-489f-b947-fab8f69ee540/create-index/serverless
    index_name = "reit4841-aistandards"

    if not pc.has_index(index_name):
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("New index created")
    else:
        print(f"Using existing index: {index_name}")
    
    index = pc.Index(index_name)

    # Initialise GoogleGeminiAI embeddings
    # Specify text_key as text to allow for referral to original text being used in response later
    print("Initializing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                            google_api_key=GOOGLE_API_KEY)

    # Initialise Vector Store
    print("Initializing vector store...")
    vector_store = PineconeVectorStore(index_name=index_name, 
                                       embedding=embeddings, 
                                       text_key="text", 
                                       pinecone_api_key=PINECONE_API_KEY)

    # Initialise document organiser
    print(f"Loading PDF documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks")

    print("Generating unique IDs for chunks...")
    uuids = [str(uuid4()) for _ in range(len(chunks))]

    print("Uploading documents to Pinecone...")
    vector_store.add_documents(documents=chunks, ids=uuids)
    print("Document ingestion completed successfully!")

if __name__ == "__main__":
    main()