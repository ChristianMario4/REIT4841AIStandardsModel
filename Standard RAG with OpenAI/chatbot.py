from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage
from pinecone import Pinecone
from pinecone.models import ServerlessSpec
import asyncio
import nest_asyncio
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


index = pc.Index(index_name)

# initialise the model, setting it to have 0 creativity in response (temp), no response size limit, reasonable timeout


# Configuration for embeddings and retrieval
num_results = 5
simscore_threshold = 0.7

# Fix to resolve re-current re-initialisation. Cache_resource makes sure this only occurs once.
@st.cache_resource # Claude recommended bug fix for syncing error thrown by Streamlit
def initialize_embeddings_and_retriever():
    
    """Setting asyncio event loop inside initialization so they run on the same thread"""
    nest_asyncio.apply() # Claude recommended bug fix for syncing error thrown by Streamlit
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Remaining code generated user
    # Initialise GoogleGeminiAI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # 1024 dimensions
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT"
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
        model="gemini-2.0-flash-exp",
        temperature=0,
        max_tokens=None,
        timeout=30,
        max_retries=3,
        google_api_key=GOOGLE_API_KEY
    )
    
    return llm

llm = initialize_llm()

# Function to get response from RAG system (Fixed by Claude for Streamlit)
def get_rag_response(message, chat_history):
    try:
        # Get the retriever (this will use cached version after first call)
        retriever = initialize_embeddings_and_retriever()
        
        # Retrieve the relevant chunks based on the question asked
        docs = retriever.invoke(message)
        
        # Debugging chunk recommended by Claude to verify contents
        st.write("### 🔍 Debug - Retrieved Chunks:")
        for i, doc in enumerate(docs, 1):
            st.write(f"**Chunk {i}** (Score: {doc.metadata.get('score', 'N/A')})")
            st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
            st.write(f"Page: {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))}")
            st.write(f"Content preview: {doc.page_content}")
            st.write(f"**Full metadata:** {doc.metadata}")
            st.write("---")

        # Add all the chunks to 'knowledge'
        knowledge = ""
        for doc in docs:
            knowledge += f"Chunk Page Content: {doc.page_content}\n\n"
            knowledge += f"Document Metadata: {doc.metadata}\n\n"

        # Create conversation history string in Streamlit UI
        history_str = ""
        for human_msg, ai_msg in chat_history:
            history_str += f"Human: {human_msg}\nAssistant: {ai_msg}\n"

        # Make the call to the LLM (including prompt)
        rag_prompt = f"""
        You are a helpful assistant that answers questions using ONLY the provided context below.
        You DO NOT have access to any other knowledge or information beyond what is explicitly provided in the "Knowledge" section.
        You must STRICLY use ONLY information from the knowledge section to answer questions. 
        Do not use any external knowledge, prior training data, or general knowledge.
        If the answer is not explicitly in the knowledge section, you must state: "I cannot answer this question based on the provided documents."
        
        If you can answer the questions based solely on the Knowledge section, proceed with the next isntructions:

        When you provide your response, can you format it so that:
        1. Your response
        2. A a new line with "---"
        3. A list of the similar chunks that you retrieved separated into dot points, stating the documents that they came from, the text, and the page label.
        Can you ensure that the list items are the same amount as the size of {len(docs)}. 
        
        With the overall list appearing as a numbered list ordered by the score metadata (each of the different texts):
        1. **Document Name:** [the filename after dat/]
        - **Page Number:** [page_label from metadata]
        - **Text:** "[the excerpt from page_content]"
        - **Score:** [score from metadata]
        
        2. **Document Name:** [the filename after dat/]
        - **Page Number:** [page_label from metadata]
        - **Text:** "[the excerpt from page_content]"
        - **Score:** [score from metadata]
        
        (continue for all {len(docs)} chunks)
        
        CRITICAL INSTRUCTIONS:
        - You must list exactly {len(docs)} sources using a numbered list (1,2,3....) ONLY if you can answer the question
        - If you cannot answer the question based on the Knowledge section, there should be no list of retrieved chunks at all
        - You should NEVER alter the contents of the excerpt from page_content when listing out the retrieved chunks
        - When answering a question, use only excerpts in page_content without modifying them. Ensure your response reads naturally
        and integrates the excerpts smoothly, rather than listing them in order. Do not draw your own conclusions beyond what the excerpts
        support
        - Do not add information from your own conclusions beyond what the excerpts explicitly support
        - The list of retrieved chunks should be in order of score
        - If the Knowledge section does not contain enough information to fully answer the question, state what information is missing
        
        
        The question: {message}

        Conversation history: {history_str}

        Knowledge: {knowledge}
        
        
        FINAL INSTRUCTION: Review the Knowledge Section above. 
        Does it contain substantive text content that answers the question? 
        If NO, respond: "I cannot answer this question based on the provided documents."
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
            chat_history = [] 

            # Grab chat history from 
            for i, msg in enumerate(st.session_state.messages[:-1]):
                if msg["role"] == "user":
                    if (i + 1 < len(st.session_state.messages)) and st.session_state.messages[i + 1]["role"] == "assistant":
                        chat_history.append((msg["content"], st.session_state.messages[i + 1]["content"]))
            
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
        st.markdown("Thomas Janssen - https://www.youtube.com/watch?v=A3WKdt_MNZQ&t=6s")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        # Claude assisted code development for resolving Streamlit not refreshing
        if st.button("Reset Retriever Settings"):
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()