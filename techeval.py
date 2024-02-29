import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core import load_index_from_storage,StorageContext
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

# global
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
from glob import glob


st.set_page_config(page_title="Chat with AI research Scientist, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
Settings.chunk_size = 512
Settings.llm= Gemini(model="models/gemini-pro")


GOOGLE_API_KEY = st.secrets.GOOGLE_API_KEY #For streamlit cloud only

st.title("Chat with the AI Scientist, powered by LlamaIndex 💬🦙")

         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Technical Interviews for data scientist roles!"}
    ]

@st.cache_resource(show_spinner=False)
def ingest_load_data():
    index = None
    with st.spinner(text="Loading and indexing the interviews – hang tight! This should take 1-2 minutes."):
        input_files = glob("./*.mp3")
        print("input files is:", input_files)
        storage_context = StorageContext.from_defaults()
        if len(input_files)>=1:
            reader = SimpleDirectoryReader(input_files=input_files, recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(llm=Gemini(model="models/gemini-pro", temperature=0.5), 
                                                           embed_model=GeminiEmbedding(model_name="models/embedding-001")
,           system_prompt="""You are a seasoned AI research scientist with over 20 years of experience 
                            and your job is to answer technical questions about interviews. If the questions is not included in the context, 
                            please provide an answer based on general knowledge."""
)
            index = VectorStoreIndex.from_documents(docs, service_context=service_context, storage_context=storage_context, embed_model=GeminiEmbedding(model_name="models/embedding-001")
)
            storage_context.persist(persist_dir="qna")
    if index is None:
        storage_context = StorageContext.from_defaults(persist_dir="qna")
        index = load_index_from_storage(storage_context, embed_model=GeminiEmbedding(model_name="models/embedding-001")
)
        
    return index


# @st.cache_data(show_spinner=False)
# def load_data():
#     #service_context = ServiceContext.from_defaults(llm=Gemini(model="models/gemini-pro", temperature=0.5, system_prompt="You are a seasoned AI research scientist with over 20 years of experience and your job is to answer technical questions about interviews. If the questions is not included in the context, please provide an answer based on general knowledge."))
#     storage_context = StorageContext.from_defaults(persist_dir="questions")
#     try:
#         index = load_index_from_storage(storage_context)
#     except ValueError as e:
#         print(e)
#         index = None

#     if index is None:
#         index = ingest_data()

#     return index



#print(f" The index is : {index}")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        index = ingest_load_data()
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history