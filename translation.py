import os
from transformers import MarianMTModel, MarianTokenizer
from file_reading import extract_text

# Load translation models 
def load_translation_model(target_lang):
    model_name = {
        "en": "Helsinki-NLP/opus-mt-ar-en", 
        "ar": "Helsinki-NLP/opus-mt-en-ar"
    }[target_lang]
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Translate text
def translate_text(text, tokenizer, model):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Chunk to avoid long inputs
    translations = []
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        translations.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    return "\n".join(translations)

# Main translation function
def translate_file(file_path, target_lang):
    print(f"Translating {file_path} to {target_lang}...")
    text = extract_text(file_path)
    tokenizer, model = load_translation_model(target_lang)
    translated_text = translate_text(text, tokenizer, model)
    return translated_text
