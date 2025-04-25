import faiss
import pickle
from llama_index.core import load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine

# Function to load the FAISS index and metadata
def load_faiss_index(index_path, metadata_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

# Function to get an answer based on the query using LLaMA
def answer_query(index_path, metadata_path, query, llama_model_path):
    # Load FAISS index and metadata
    index, metadata = load_faiss_index(index_path, metadata_path)
    
    # Load LLM (update this if using local model)
    llama = OpenAI(model_path=llama_model_path)

    # Create the query engine manually
    query_engine = RetrieverQueryEngine.from_args(index.as_retriever(), llm=llama)

    # Query
    response = query_engine.query(query)
    
    return response.response
