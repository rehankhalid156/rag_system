import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch

# Load FAISS index and metadata
print("Loading sentence-transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print(" Loading zero-shot classification model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to use", device)

print(" Loading T5-small model for structured generation...")
from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)


# Load saved index and metadata
index = faiss.read_index("data/ibm_argument_index.faiss")
metadata = pd.read_csv("data/ibm_argument_metadata.csv")

def classify_argument(text):
    labels = ["ETHICAL", "LOGICAL", "EMOTIONAL"]
    result = classifier(text, labels)
    return result["labels"][0]

def generate_summary_t5(prompt):
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = t5_model.generate(input_ids, max_length=100)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def retrieve_classify_generate(query, top_k=5):
    print(f"\n Top {top_k} results for:\n {query}")
    
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        row = metadata.iloc[idx]
        topic = row["topic"]
        argument = row["argument"]
        source = row["source_title"]
        url = row["source_url"]
        label = classify_argument(argument)
        
        print(f"\nTopic: {topic}")
        print(f" Argument: {argument}")
        print(f" Type: {label}")
        print(f" Source: {source}")
        print(f" URL: {url}")
        
        results.append(f"{argument} (from {source})")

    print("\n Generating structured summary using T5...\n")
    summary_input = f"Generate a structured summary of arguments: {' '.join(results)}"
    print(generate_summary_t5(summary_input))

# Main loop
while True:
    user_input = input("\nEnter a debate topic (or 'exit' to quit):\n> ")
    if user_input.lower() == "exit":
        break
    retrieve_classify_generate(user_input)
