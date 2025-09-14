"""
Simple RAG System for Pizza Restaurant Reviews

This module implements a Retrieval-Augmented Generation system that allows users
to query restaurant reviews using natural language. The system combines vector
search with large language models to provide contextual answers.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize the language model
# Uses Ollama's local LLM instance with the Llama 3.2 model
model = OllamaLLM(model="llama3.2")

# Define the prompt template for the RAG system
# The template instructs the model to act as a restaurant expert and includes
# placeholders for retrieved reviews and user questions
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews:{reviews}

Here is the question to answer: {question}
"""

# Create a structured prompt template from the string template
prompt = ChatPromptTemplate.from_template(template)

# Create the processing chain that connects the prompt template to the model
# Uses LangChain's pipe operator to create a sequential processing pipeline
chain = prompt | model

# Main interactive loop for handling user queries
while True:
    print("\n\n====================================")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    
    # Exit condition for the interactive loop
    if question == "q":
        break

    # Retrieve relevant reviews using vector similarity search
    reviews = retriever.invoke(question)
    
    # Process the question with retrieved context through the LLM chain
    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })
    
    # Display the generated response
    print(result)