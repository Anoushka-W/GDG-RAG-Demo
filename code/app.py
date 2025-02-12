import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import nltk
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------- 1 - Data Ingestion ----------------------------

# Function to load the file, split it into chunks, and add them to the vector store
def add_to_vector_store(file, vector_store, chunk_size=1000, chunk_overlap=200):
    if file:
        # Use tempfile because Langchain Loaders only accept a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_file_path = tmp.name

        # Use Langchain Loaders to load the file into a Document object (which stores page content and metadata)
        if file.type == "application/pdf":
            loader = PyPDFLoader(file_path = tmp_file_path)
        elif file.type == "application/json":
            loader = JSONLoader(file_path = tmp_file_path, jq_schema=".", text_content=False)
        elif file.type == "text/markdown":
            loader = UnstructuredMarkdownLoader(file_path = tmp_file_path)        
        else:
            loader = TextLoader(file_path = tmp_file_path)

        data = loader.load()

        # Replace temporary file name with original file name in documents' metadata
        for document in data:
            document.metadata["source"] = file.name

        print(f"Loaded {len(data)} documents from {file.name}")
        # Use Langchain Text Splitter to split the document into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                  chunk_overlap=chunk_overlap,
                                                  add_start_index=True,  # track index in original document
                                                )
        chunked_data = splitter.split_documents(data)
        
        print(f"Chunked {file.name} into {len(chunked_data)} pieces")

        # Upload the chunked data to the ChromaDB collection
        uuids = [file.name + str(uuid4()) for _ in range(len(chunked_data))]
        vector_store.add_documents(documents=chunked_data, ids=uuids)

        print(f"Uploaded {file.name} to ChromaDB")
        
        # Delete the temporary file
        tmp.close()
        os.unlink(tmp_file_path)
  
# ---------------------------- 2 - Query Processing ----------------------------

def rewrite_query(user_query, llm):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable assistant that generates a well-formed document to answer a question."),
        ("human", f"""Create a document that answers the below question.
        
The document should:
- Be well-structured and informative
- Provide a detailed response that directly answers the question
- Maintain clarity and coherence

Question:
```
{user_query}
```

Generated Document:
""")
    ])

    # Generate the hypothetical document using the LangChain model
    chain = prompt | llm
    ai_message = chain.invoke({
        "user_query": user_query,
    })

    # Append the original query to ensure context is retained
    hypothetical_document = user_query + "\n" + ai_message.content.strip()

    print("Original query:", user_query)
    print("Generated Hypothetical Document:", hypothetical_document)

    return hypothetical_document

# Function to handle the user input submission
def chat(user_query, llm, retriever, conversation_history):   
    # Rewrite query to utilize Hypothetical Document Embeddings (HyDE)
    rewritten_query = rewrite_query(user_query, llm)
        
    # Retrieve relevant context for the rewritten query from the vector database
    retrieved_documents = retriever.invoke(rewritten_query)

    print("Number of retrieved documents:", len(retrieved_documents))

    # Extract the text content of the retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_documents])

    print("\n Retrieved context: ```", context, "```")

    # Create a list of LangChain messages from the conversation history (limit to last 4 messages - starts with human message, ends with AI message)
    messages = [HumanMessage(msg['content']) if msg['role'] == 'user' else AIMessage(msg['content']) for msg in conversation_history[-4:]]
    
    
    # Add system message and human message 
    messages.insert(0, SystemMessage("Answer the following question using the retrieved context. Provide a concise and informative answer that directly addresses the user's question. If the provided context does not contain the answer to the user's question, simply reply: 'I couldn't find the necessary information to answer your question. Please update your prompt or provide more documents that may be relevant to your question.' Use a maximum of three sentences to answer the question."))
    messages.append(HumanMessage(f"""Question: 
```
{user_query}
```

Context:
```
{context}
```

Answer:
"""
))  

    print("\nMessages:", messages)

    # Generate the response from the model
    return llm.stream(messages)

# ---------------------------- Initialization ----------------------------
print("Initializing...")

# Initialize session state for uploaded files, model, top_k and messages
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
if 'model' not in st.session_state:
    st.session_state['model'] = "gemma2:2b"
if 'top_k' not in st.session_state:
    st.session_state['top_k'] = 3  
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize Chat Ollama model
llm = ChatOllama(
    model = st.session_state["model"],
    temperature = 0.8
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Initialize chromadb 
vector_store = Chroma(
    collection_name="vault",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


# Use the vector store as a retriever
retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": st.session_state['top_k']}
)

# ---------------------------- Streamlit UI ----------------------------
# # 1. DISPLAY CHAT MESSAGES
st.title("Vault App")

st.markdown("Welcome to the Vault App! Upload a file and ask a question to retrieve relevant context from the uploaded documents.")

# Go through the chat history and display the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. SIDEBAR
# Uploading files
st.sidebar.header("Upload a file")
uploaded_files = st.sidebar.file_uploader("Choose a file", 
                                          type=["pdf", "txt", "json", "md"],
                                          accept_multiple_files=True)

# If files have not been loaded into the ChromaDB collection, load them
if uploaded_files:
    new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
    if new_files:
        for new_file in new_files:
            add_to_vector_store(new_file, vector_store)
        st.session_state.uploaded_files.extend(new_files)

# Settings
st.sidebar.header("Settings")
st.session_state["model"] = st.sidebar.selectbox("Select Model", ["gemma2:2b", "gemma2"], index=0) # Model to use
st.session_state["top_k"] = st.sidebar.slider("Top K Context", 1, 5, value=st.session_state.top_k)  # Top K context to retrieve

# Toggle to reset conversation
st.sidebar.button("Reset Conversation", on_click= lambda: st.session_state.update(messages=[]))

# Toggle to clear uploaded files
def clear_uploaded_files():
    st.session_state.update(uploaded_files=[])
    vector_store.reset_collection()
    print("Cleared vault documents")
    st.toast("**CLEARED VAULT DOCUMENTS**", icon = "🚨")

st.sidebar.button("Clear Vault Documents", on_click = clear_uploaded_files)

# 3. USER INPUT
# When the user enters a query
user_query = st.chat_input("Enter your message")
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = chat(user_query = user_query, 
                    llm = llm, 
                    retriever = retriever,
                    conversation_history = st.session_state['messages'][:-1:])
        
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
