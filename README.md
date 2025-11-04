# REIT4841AIStandardsModel
OpenAI Chatbot Designed to Respond to Prompts Regarding AI Standards

This repository contains all of the relevant code and files required to make the chatbot operational. 
If you only want to access the chatbot, please click this link: https://reit4841aistandardsmodel-5sekmzxy3r2r7nfbmjxph8.streamlit.app/

Otherwise, the following details the contents of this repository and the purpose of each of the files contained. 
All of these key files are contained within the Standard RAG with OpenAI folder. 
NOTE: The original locally hosted chatbot made use of OpenAI large language models that were locally downloaded.
      However, the current (cloud) version of the software uses Gemini (gemini-2.0-flash-exp).

*This folder contains:*
data: A folder containing the whole corpus of documents
ingest_database.py: A Python script responsible for processing the PDFs inside the corpus of documents, 
                    splitting them into chunks, vectorizing them using a specified embedding model 
                    (GoogleGenerativeAIEmbeddings) and storing it within the vector database. 
chatbot.py: The main Python script containing the chatbot logic.

*Outside of the folder:*
requirements.txt: Contains dependenceies to required to execute the scripts

An explanation of the purpose of each of the scripts, and how to make changes if desired can be seen in the following sections:

**ingest_database.py:**
This script is responsible for processing all of the documents in the corpus (presumed to be PDFs with embedded text).
This script is not touched by the hosting provider (Streamlit), and should only be manually triggered by authorised users. 

Lines of code that can be adjusted here if desired is the chunk_size and chunk_overlap for the text splitter.
To deploy these changes, a local copy of the repository will need to be made, and the relevant API key access would be required. 
For the API keys, contact *c.almario@student.uq.edu.au*.

Instructions to execute script:
1. Make a local copy of the repository
2. Optional: Make a virtual environment to install and store dependencies required to execute the script (If not completed, just download the dependencies locally using pip)
3. If you installed a virtual environment, activate it, otherwise skip this step
4. Make relevant changes to the script and save (ensuring that you have replaced the API key placeholders with the actual API keys)
5. Using Terminal, execute the ingest_database.py using python or python3

**chatbot.py:**
This script contains all of the chatbot logic, the Streamlit UI generation code, as well as integrations with the various software components used.
Unlike the ingest_database.py, this script does not require the user to have the API keys, as these are stored on Streamlit under their secured
secrets functionality. Hence, Streamlit is able to execute this code with no issues. 
Streamlit uses this script as its default file when the GitHub Repository is connected to
Streamlit (permissions had to be relaxed for Streamlit to be able to have full access to this public repository). 
When visiting the chatbot URL: https://reit4841aistandardsmodel-5sekmzxy3r2r7nfbmjxph8.streamlit.app/, Streamlit runs this script. 
Anytime a new change is pushed to the repository, Streamlit automatically updates the chatbot to match the script on GitHub.
However, due to the caching functionality implemented, the users may still need to use the clear cache button to have the updated version of the script on the Chatbot URL. 
