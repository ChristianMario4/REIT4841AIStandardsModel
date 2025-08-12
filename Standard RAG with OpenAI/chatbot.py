from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage
from pinecone import Pinecone
from pinecone.models import ServerlessSpec
import streamlit as st

# Initialise UI
st.set_page_config(
    page_title="AI Standards RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
    
st.title("🤖 AI Standards RAG Chatbot")
st.markdown("Ask questions about AI standards based on your knowledge base!")

# configuration
DATA_PATH = r"data"
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
index_name = "reit4841-aistandards"

# Initialise Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialise the index for Pinecone DB -> code from Pinecone Docos
# http://app.pinecone.io/organizations/-OWEHQS7XzaN5h_xkrqU/projects/96720a04-95b1-489f-b947-fab8f69ee540/create-index/serverless

if not pc.has_index(index_name):
    st.info(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1024,  # Match your existing index dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    st.success("New index created")
    
st.info(f"Using existing index: {index_name}")

index = pc.Index(index_name)

# initialise the model, setting it to have 0 creativity in response (temp), no response size limit, reasonable timeout


# Configuration for embeddings and retrieval
num_results = 5
simscore_threshold = 0.5

# Fix to resolve re-current re-initialisation. Cache_resource makes sure this only occurs once.
@st.cache_resource
def initialize_embeddings_and_retriever():
    """Initialize embeddings and retriever - cached to avoid re-initialization"""
    # Initialise GoogleGeminiAI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # 1024 dimensions
        google_api_key=GOOGLE_API_KEY
    )
    
    # Initialise and connect vector store
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        text_key="text"
    )
    
    # Set up the vectorstore to be the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": num_results, "score_threshold": simscore_threshold}
    )
    
    return retriever

# Initialize LLM
@st.cache_resource
def initialize_llm():
    """Initialize OpenAI LLM - cached to avoid re-initialization"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        max_tokens=None,
        timeout=30,
        max_retries=3,
)

llm = initialize_llm()

# Function to get response from RAG system (Fixed for Streamlit)
def get_rag_response(message, chat_history):
    try:
        # Get the retriever (this will use cached version after first call)
        retriever = initialize_embeddings_and_retriever()
        
        # Retrieve the relevant chunks based on the question asked
        docs = retriever.invoke(message)

        # Add all the chunks to 'knowledge'
        knowledge = ""
        for doc in docs:
            knowledge += doc.page_content + "\n\n"

        # Create conversation history string
        history_str = ""
        for human_msg, ai_msg in chat_history:
            history_str += f"Human: {human_msg}\nAssistant: {ai_msg}\n"

        # Make the call to the LLM (including prompt)
        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.

        The question: {message}

        Conversation history: {history_str}

        The knowledge: {knowledge}
        """

        # Invoke the response
        response = llm.invoke(rag_prompt)
        
        # If response is an AIMessage, access the content
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
def main():
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask me anything about AI standards..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Create a placeholder for response
            message_placeholder = st.empty()
            
            # Get chat history for context (excluding current message)
            chat_history = [(msg["content"], st.session_state.messages[i+1]["content"]) 
                          for i, msg in enumerate(st.session_state.messages[:-1]) 
                          if msg["role"] == "user" and i+1 < len(st.session_state.messages)]
            
            # Get response from RAG system
            with st.spinner("Thinking..."):
                full_response = get_rag_response(prompt, chat_history)
            
            # Display the response
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("This RAG LLM was developed for REIT4841 by Christian Almario")
        st.markdown("This chatbot uses:")
        st.markdown("- **Pinecone** for vector storage")
        st.markdown("- **Google Gemini** for LLM")
        st.markdown("- **RAG** for knowledge-based answers")
        
        st.markdown("### Settings")
        st.markdown(f"**Results per query:** {num_results}")
        st.markdown(f"**Similarity threshold:** {simscore_threshold}")

        st.markdown("### Acknowledgements")
        st.markdown("The following are inspirations for the development of this chatbot")
        st.markdown("Thomas Janssen - ")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()