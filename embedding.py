import os
from nomic import embed
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model (offline)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Works offline

def build_faiss_index(chunks, save_path):
    """
    Builds a FAISS index from a list of chunks using local embeddings.

    Args:
        chunks (List[dict]): Each dict must contain 'text', 'source', 'page', 'chunk'.
        save_path (str): Path to save FAISS index and metadata.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract text and metadata
    texts = [chunk["text"] for chunk in chunks]
    metadata = [{"source": c["source"], "page": c["page"], "chunk": c["chunk"], "text": c["text"]} for c in chunks]

    print("Generating embeddings using local sentence-transformers model...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, os.path.join(save_path, "faiss_index.index"))
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index and metadata saved to {save_path}")
