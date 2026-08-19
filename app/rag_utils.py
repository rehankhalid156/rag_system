"""
rag_utils.py
-------------
Shared core logic for the Open-Domain Debater Assistant.

Both app.py (the Streamlit web UI) and cli_demo.py (a terminal version,
useful for quick testing without launching a browser) import from this
file. This is the ONE place model-loading, retrieval, classification,
and summarization logic lives — earlier versions of this project had
this same code copy-pasted across three files, which made bugs easy to
introduce and hard to fix everywhere at once.
"""

import os

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------------------------------------------------
# Paths & config
# ----------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(DATA_DIR, "ibm_argument_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ibm_argument_metadata.csv")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLASSIFIER_MODEL_NAME = "facebook/bart-large-mnli"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"

CLASSIFICATION_LABELS = ["ETHICAL", "LOGICAL", "EMOTIONAL"]
MAX_SUMMARY_INPUT_CHARS = 3500   # keeps input under BART's 1024-token limit
TOP_K = 5

# Below this similarity, we tell the user honestly that we probably don't
# have good matches, instead of confidently showing irrelevant results.
# (FAISS always returns *something* — it never says "no match found" —
# so this threshold is what makes that distinction visible to the user.)
LOW_RELEVANCE_THRESHOLD = 0.25


# ----------------------------------------------------------------------
# Custom exceptions — used so app.py / cli_demo.py can show a clear,
# friendly message instead of letting a raw traceback crash the app.
# ----------------------------------------------------------------------
class DataNotFoundError(Exception):
    """Raised when the FAISS index or metadata CSV is missing."""
    pass


class ModelLoadError(Exception):
    """Raised when a Hugging Face model fails to download/load."""
    pass


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_data():
    """Loads the FAISS index + metadata CSV, with a clear error if missing."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise DataNotFoundError(
            f"Dataset files not found at:\n  {INDEX_PATH}\n  {METADATA_PATH}\n\n"
            f"Run 'python app/build_index.py' first to build the dataset."
        )
    metadata = pd.read_csv(METADATA_PATH)
    index = faiss.read_index(INDEX_PATH)
    return metadata, index


def load_models():
    """Loads all three AI models, with a clear error if a download fails."""
    try:
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL_NAME)
        summarizer = pipeline("summarization", model=SUMMARIZER_MODEL_NAME, device=-1)
    except Exception as e:
        raise ModelLoadError(
            "Failed to load one of the AI models. This usually means either "
            "there's no internet connection (models download from Hugging "
            f"Face on first run) or the download was interrupted.\n\nDetails: {e}"
        )
    return embedder, classifier, summarizer


# ----------------------------------------------------------------------
# Core RAG steps
# ----------------------------------------------------------------------
def retrieve_arguments(embedder, index, metadata, query, top_k=TOP_K):
    """Returns a list of dicts: topic, argument, stance, source, similarity."""
    if not query or not query.strip():
        return []

    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue  # FAISS can return -1 if fewer than top_k vectors exist
        row = metadata.iloc[idx]
        # NOTE: build_index.py uses faiss.IndexFlatIP on L2-normalized
        # vectors, so `dist` here IS the cosine similarity already
        # (higher = more similar). Using "1 - dist" (correct for an
        # IndexFlatL2 distance index) was inverting this — it made
        # irrelevant results score HIGHER than relevant ones.
        results.append({
            "topic": row.get("topic", ""),
            "argument": row.get("argument", ""),
            "stance": str(row.get("stance", "")).upper(),
            "source": row.get("source_title", "unknown"),
            "similarity": float(dist),
        })
    return results


def classify_argument(classifier, argument_text):
    """Returns (label, confidence) using the REAL model confidence score."""
    if not argument_text or not argument_text.strip():
        return "UNKNOWN", 0.0
    result = classifier(argument_text, CLASSIFICATION_LABELS)
    return result["labels"][0], round(result["scores"][0], 3)


def calculate_stance_strength(similarity, classification_confidence):
    """Blends real retrieval similarity with real classification confidence."""
    return round((0.7 * similarity + 0.3 * classification_confidence), 3)


def generate_structured_summary(summarizer, topic, arguments_text):
    """Summarizes a block of arguments with BART. Returns a friendly message
    instead of crashing if there's nothing to summarize or generation fails."""
    if not arguments_text or not arguments_text.strip():
        return "No arguments were available to summarize."

    combined_text = f"Debate topic: {topic}. Arguments: {arguments_text}"
    combined_text = combined_text[:MAX_SUMMARY_INPUT_CHARS]

    try:
        summary = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Could not generate a summary due to an error: {e}"