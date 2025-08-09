import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Load models
print("🧠 Loading sentence-transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🧠 Loading zero-shot classification model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

device = torch.device("cpu")
print("Device set to use", device)

print("🧠 Loading T5-small model for structured generation...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Load metadata and FAISS index
metadata = pd.read_csv("data/ibm_argument_metadata.csv")
index = faiss.read_index("data/ibm_argument_index.faiss")

# Function to classify argument type
def classify_argument(text):
    labels = ["ETHICAL", "LOGICAL", "EMOTIONAL"]
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]

# Function to generate structured summary using T5
def generate_structured_summary(topic, arguments):
    prompt = f"Summarize arguments for the topic: '{topic}'\nArguments:\n{arguments}"
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    summary_ids = t5_model.generate(inputs.input_ids, max_length=256, num_beams=5, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return sentence_bleu([reference_tokens], candidate_tokens)

# Main loop for retrieval, classification, and generation
def retrieve_classify_generate(query, reference_summary=None, top_k=5):
    print(f"\n🔍 Top {top_k} results for:\n➡️ {query}")
    
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    args = []

    for i in range(top_k):
        idx = I[0][i]
        similarity = 1 - D[0][i]
        row = metadata.iloc[idx]

        argument_text = row["argument"]
        topic = row["topic"]
        source = row["source_title"]
        url = row["source_url"]
        label = classify_argument(argument_text)

        stance_score = round((0.7 * similarity + 0.3 * 0.9), 3)  # Stance strength score

        args.append(argument_text)

        print(f"\n🟩 Topic: {topic}")
        print(f"🗣️ Argument: {argument_text}")
        print(f"🏷️ Type: {label}")
        print(f"📈 Stance Strength Score: {stance_score}")
        print(f"📚 Source: {source}")
        print(f"🔗 URL: {url}")

    print("\n🧠 Generating structured summary using T5...\n")
    structured_summary = generate_structured_summary(query, "\n".join(args))
    print(f"Structured Summary: {structured_summary}")

    # Evaluate if reference summary is provided
    if reference_summary:
        print("\n🔍 Evaluating summary quality with ROUGE and BLEU:")
        rouge_scores = calculate_rouge(reference_summary, structured_summary)
        print(f"ROUGE Scores: {rouge_scores}")
        
        bleu_score = calculate_bleu(reference_summary, structured_summary)
        print(f"BLEU Score: {bleu_score}")

# Run loop
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a debate topic (or 'exit' to quit):\n> ")
        if user_input.lower() == "exit":
            break
        reference_summary = "The use of plastic bags causes environmental harm due to pollution and wildlife impact."
        retrieve_classify_generate(user_input, reference_summary=reference_summary)
