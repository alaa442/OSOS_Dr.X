RAG-based Q&A System for Document Retrieval and Summarization

Overview
This project implements a Retrieval-Augmented Generation (RAG)-based question and answer system designed to answer questions based on document contents, including summarization and translation steps. The system uses vector embeddings to store and retrieve relevant information, and then a large language model (LLM) is utilized to generate coherent answers based on this retrieved data.

Methodology
The project is divided into several stages, each of which contributes to the goal of answering questions based on publication content:

Document Preprocessing: The system reads and processes various types of documents (e.g., .txt, .docx, .pdf, etc.) and extracts the text.

Embedding Generation:
We generate vector embeddings for the extracted text using an embedding model.
These embeddings represent the semantic meanings of the documents and enable efficient retrieval for answering queries.
The Faiss library is used to index the embeddings, making the retrieval of the most relevant text segments possible.

Translation: If documents or questions are in multiple languages, we translate them into a common language to ensure consistent processing across different languages. This is done using a translation model of your choice.
Summarization: Since the documents can be lengthy, they are chunked into smaller pieces, and each piece is summarized using an automatic summarization model. The summaries are then stored for future retrieval.

RAG-based Query Answering:
The system answers questions by using Retrieval-Augmented Generation (RAG).
The query is embedded and matched to the relevant chunks of text from the vector database.
The retrieved chunks are used as context for a language model (LLM) to generate the final answer.
Performance Tracking: During the embedding generation, translation, summarization, and RAG processes, the system tracks and reports the tokens per second (TPS) processed by the LLM. This helps measure the efficiency of the system.

LLM Used
The project utilizes the Llama language model, a state-of-the-art model known for its efficiency and ability to handle various NLP tasks such as text generation, summarization, and question answering.

Model Used: Llama (loaded via the llama_index package).
Model Source: The model is obtained from the LLaMA library (e.g., through Hugging Face or local models).
The Llama model is fine-tuned for the specific task of handling long-form document queries, ensuring that it can generate accurate and relevant responses based on the retrieved document fragments.

Embedding Model
For generating embeddings, we utilize a transformer-based embedding model, such as Sentence Transformers or another model capable of generating dense vector representations for documents.

Embedding Model: Sentence Transformers or equivalent transformer model (you can specify the exact model you are using).

Library: We use the Faiss library to index and search embeddings efficiently.

The embedding model converts text data into vectors of fixed size, capturing the semantic meaning of the text. These embeddings are stored in a FAISS index to enable efficient retrieval during the RAG process.

How to Run
1- Please download llama-2-7b.Q4_0.gguf folder inside "models/" folder
2- Install the required dependencies by running the following command: pip install -r requirements.txt
3- Running the System:
  You can run the system through the main script or Jupyter Notebook:
  The system will read documents, generate embeddings, summarize text, translate, and then respond to a query by retrieving relevant information from the indexed documents and generating a response with the Llama model.
4- Customization
Documents: Replace the input_directory with the path to your documents.


Significant Discoveries
Vector Database Efficiency: Using FAISS to index the document embeddings enabled fast and efficient retrieval, even for large datasets.
Summarization Performance: The summarization model was able to effectively condense large documents into smaller, more manageable pieces while preserving key information. However, some document types, like PDFs, require more preprocessing.
Translation Challenges: Handling multilingual content required careful translation handling, but using the translation models allowed us to standardize inputs and produce more consistent answers.
LLM Query Handling: The Llama model provided coherent and contextually relevant answers based on the retrieved document chunks. The inclusion of token tracking helped monitor performance during the querying phase.
