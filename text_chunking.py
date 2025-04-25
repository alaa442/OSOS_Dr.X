import tiktoken

def chunk_file_text(text, filename, max_tokens=500, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    chunk_num = 1

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunk_data = {
            "source": filename,
            "page": 1,  # Placeholder, update if real page info is available
            "chunk": chunk_num,
            "text": chunk_text
        }
        chunks.append(chunk_data)

        chunk_num += 1
        start += max_tokens - overlap

    return chunks
