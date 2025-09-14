"""
Vector Database Module for Restaurant Review RAG System

This module handles the creation and management of a vector database for restaurant
reviews, including document processing, embedding generation, and similarity search
retrieval functionality.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load restaurant reviews dataset from CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize embedding model for vector representations
# Uses Ollama's mxbai-embed-large model for high-quality embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define persistent storage location for the vector database
db_location = "./chroma_langchain_db"

# Check if database already exists to avoid re-processing documents
add_document = not os.path.exists(db_location)

# Process CSV data into LangChain Document objects if database doesn't exist
if add_document:
    documents = []
    ids = []

    # Convert each review row into a Document object
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],  # Combine title and review text
            metadata={"rating": row["Rating"], "date": row["Date"]},  # Store additional context
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Initialize Chroma vector database with persistent storage
# By default, Chroma uses cosine similarity for vector comparisons
vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

# Add documents to vector store on first run
if add_document:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever interface for similarity search queries
# Returns top 5 most relevant documents using cosine similarity
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}   
)