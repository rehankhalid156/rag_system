import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
# Set custom background color and title styling
st.set_page_config(page_title="Open-Domain Debater Assistant", page_icon="🧠")

# Custom CSS to style the background and input
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
        color: #333;
    }
    .stTextInput>div>div>input {
        background-color: #d0e6ff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stText {
        font-size: 18px;
        color: #333;
    }
    .stMarkdown {
        font-size: 18px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
st.title("🧠 Open-Domain Debater Assistant")

st.write("""
This app lets you input a debate topic, retrieves relevant arguments, 
classifies them, and generates structured summaries (Point/Counterpoint).
""")

# Load the models only once
@st.cache_resource
def load_models():
    print("🧠 Loading sentence-transformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("🧠 Loading zero-shot classification model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    device = torch.device("cpu")

    print("🧠 Loading T5-small model for structured generation...")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    # Load the FAISS index and metadata
    metadata = pd.read_csv("data/ibm_argument_metadata.csv")
    index = faiss.read_index("data/ibm_argument_index.faiss")

    return embedder, classifier, t5_tokenizer, t5_model, metadata, index

# Cache the models and resources
embedder, classifier, t5_tokenizer, t5_model, metadata, index = load_models()

# Function to classify argument type
def classify_argument(argument):
    labels = ["ETHICAL", "LOGICAL", "EMOTIONAL"]
    result = classifier(argument, labels)
    return result["labels"][0]

# Function to generate structured summary using T5
def generate_structured_summary(topic, arguments):
    prompt = f"Summarize arguments for the topic: '{topic}'\nArguments:\n{arguments}"
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True)
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

# Streamlit Interface
query = st.text_input("Enter a debate topic:", placeholder="E.g. Should we ban plastic bags?")

if query:
    st.write(f"**Searching for arguments related to: {query}**")
    
    # Retrieve arguments
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), 5)

    arguments = []

    # Display the retrieved arguments
    for i in range(5):
        idx = I[0][i]
        similarity = 1 - D[0][i]
        row = metadata.iloc[idx]

        argument_text = row["argument"]
        topic = row["topic"]
        source = row["source_title"]
        url = row["source_url"]
        label = classify_argument(argument_text)

        stance_score = round((0.7 * similarity + 0.3 * 0.9), 3)  # Here you can adjust the weight if needed

        arguments.append(argument_text)

        st.write(f"🟩 **Topic:** {topic}")
        st.write(f"🗣️ **Argument:** {argument_text}")
        st.write(f"🏷️ **Type:** {label} (Confidence: 0.9)")
        st.write(f"📈 **Stance Strength Score:** {stance_score}")
        st.write(f"📚 **Source:** {source}")
        st.write(f"🔗 **URL:** {url}")
        st.write("---")

    st.write("🧠 Generating structured summary using T5...")

    # Generate the structured summary using T5
    structured_summary = generate_structured_summary(query, " ".join(arguments))
    st.write(f"**Structured Summary:**")
    st.write(structured_summary)

    # Evaluate the summary (if reference summary is provided)
    reference_summary = st.text_area("Enter a reference summary for evaluation (optional):")

    if reference_summary:
        st.write("\n🔍 Evaluating summary quality with ROUGE and BLEU:")
        rouge_scores = calculate_rouge(reference_summary, structured_summary)
        st.write(f"**ROUGE Scores:** {rouge_scores}")
        
        bleu_score = calculate_bleu(reference_summary, structured_summary)
        st.write(f"**BLEU Score:** {bleu_score}")
