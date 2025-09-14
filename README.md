# Simple RAG - Restaurant Review Query System

A Retrieval-Augmented Generation (RAG) system for querying restaurant reviews using natural language. This system combines vector search with large language models to provide contextual answers about restaurant experiences.

## Features

- **Vector Search**: Uses Chroma DB with cosine similarity for semantic search
- **Local LLM**: Integrates with Ollama's Llama 3.2 model for response generation  
- **Professional Embeddings**: Utilizes mxbai-embed-large for high-quality vector representations
- **Interactive Interface**: Command-line interface for natural language queries
- **Persistent Storage**: Maintains vector database across sessions

## Architecture

- `main.py` - Main application with interactive query interface
- `vector.py` - Vector database setup and retrieval functionality
- `realistic_restaurant_reviews.csv` - Dataset of restaurant reviews
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Ollama models: `llama3.2` and `mxbai-embed-large`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MarcosRos002/Simple-RAG.git
cd Simple-RAG
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required Ollama models:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Ask natural language questions about the restaurant:
- "How are the vegan options?"
- "What do customers think about the pizza crust?"
- "Are there any complaints about service?"

3. Type 'q' to quit

## Example Query

```
Ask your question (q to quit): How are the vegan options?

Based on the reviews provided, the vegan options at this pizza restaurant are well-regarded by many customers. Reviewers have praised the vegan cheese substitute as "good," with one reviewer even stating that their non-vegan friends love the vegan pizzas...
```

## Technical Details

- **Vector Database**: Chroma with cosine similarity search
- **Embeddings**: mxbai-embed-large via Ollama
- **LLM**: Llama 3.2 via Ollama  
- **Framework**: LangChain for RAG pipeline
- **Search Results**: Returns top 5 most relevant reviews per query

## License

MIT License

