import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
import os
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
# Use this line of code if you have a local .env file
from pinecone import Pinecone
# initialize without metadata filter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex
from IPython.display import Markdown, display
from messages import *
load_dotenv() 


# global
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

st.set_page_config(page_title="Chat with AI research Scientist, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
Settings.chunk_size = 512
Settings.llm= OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt=system_message)
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]


# Add this to streamlit configuration
#OPENAI_API_KEY = st.secrets.OPENAI_API_KEY #For streamlit cloud only
#PINECONE_API_KEY= st.secrets.PINECONE_API_KEY #For streamlit cloud only

st.title("Chat with the AI Scientist, powered by LlamaIndex 💬🦙")

         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Technical Interviews for data scientist roles!"},
    ]

#@st.cache_resource(show_spinner=False)
def get_chatengine(index_name="interview"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="window") #changed text key from window to original_text
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
    
    rerank = SentenceTransformerRerank(
    top_n = 2, 
    model = "BAAI/bge-reranker-base"
)
    
    chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    verbose=True,
    similarity_top_k = 6, 
    vector_store_query_mode="hybrid", 
    alpha=0.5,
    node_postprocessors = [postproc, rerank],
)

    return chat_engine

with st.sidebar:
    st.header("About")
    st.markdown(
        """This chatbot uses retrieval augmented generation(RAG) to get insights
        from interviews performed by consultants.
        """
    )

    st.header("Example Questions")
    for question in questions:
        st.markdown(f"- {question}")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = get_chatengine()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
