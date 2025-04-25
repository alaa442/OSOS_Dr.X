from transformers import pipeline
import os

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize long texts by chunking them into smaller pieces
def summarize_text(text, chunk_size=1024):
    paragraphs = text.split('\n')
    summaries = []

    for para in paragraphs:
        # Tokenize the paragraph and check length
        if len(para.split()) <= chunk_size:
            summary = summarizer(para, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        else:
            # Break the paragraph into chunks if it's too long
            chunks = [para[i:i + chunk_size] for i in range(0, len(para), chunk_size)]
            for chunk in chunks:
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])

    # Combine all chunk summaries into a single summary
    final_summary = ' '.join(summaries)
    return final_summary

# Save the summary to a new file with "summarized_" prefix
def save_summary(summary, file_path):
    # Get the base filename and append the prefix "summarized_"
    base_filename = os.path.basename(file_path)
    summary_filename = f"summarized_{base_filename}"
    
    # Get the directory of the original file
    directory = os.path.dirname(file_path)
    
    # Full path for the new summarized file
    new_file_path = os.path.join(directory, summary_filename)
    
    # Write the summary to the new file
    with open(new_file_path, 'w') as f:
        f.write(summary)
    
    return new_file_path

# Load text from file with proper encoding handling
def load_text(file_path):
    # Try reading the file with UTF-8 encoding first, then fallback to ISO-8859-1 if it fails
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying ISO-8859-1...")
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            return f.read()


# ROUGE Evaluation
def evaluate_summary(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    print("ROUGE Scores:")
    for metric, score in scores.items():
        print(f"{metric}: Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1: {score.fmeasure:.4f}")