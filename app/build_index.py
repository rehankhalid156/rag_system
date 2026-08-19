"""
build_index.py
----------------
Downloads the args.me debate-argument corpus (100k+ English arguments),
cleans it, generates sentence embeddings, and builds a FAISS index +
metadata CSV — ready to be used by app.py / retrieve.py.

Run this ONCE (or whenever you want to rebuild the dataset) from the
project root:

    python app/build_index.py

Output files (overwrites the old ones):
    data/ibm_argument_index.faiss
    data/ibm_argument_metadata.csv
"""

import os
import re

import numpy as np
import pandas as pd
import faiss
import ijson
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG — tweak these if you want a bigger/smaller dataset
# ----------------------------------------------------------------------
SAMPLE_SIZE = 100_000          # how many arguments to keep (100k+ requested)
MIN_ARG_CHARS = 40             # drop arguments shorter than this (junk/fragments)
MAX_ARG_CHARS = 1200           # drop arguments longer than this (rambling walls of text)
MIN_WORD_COUNT = 8             # drop arguments with fewer real words than this
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(DATA_DIR, "ibm_argument_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ibm_argument_metadata.csv")

# Path to the manually-downloaded args.me corpus (867MB JSON file).
# Downloaded from: https://zenodo.org/records/4139439
RAW_JSON_PATH = os.path.join(DATA_DIR, "raw_argsme", "args-me-1.0-cleaned.json")

# A small set of very common English words. A real debate sentence should
# contain several of these. Gibberish / keyboard-mashing / joined-word junk
# usually contains none, which is what makes this check effective.
COMMON_ENGLISH_WORDS = {
    "the", "is", "are", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "that", "this", "it", "as", "be", "was", "were", "have",
    "has", "not", "we", "you", "they", "a", "an", "because", "so", "than",
    "should", "would", "can", "will", "which", "their", "there", "these",
}

# NOTE: we deliberately do NOT use the `better_profanity` library here.
# It does expensive leetspeak-normalization on every word of every text,
# which caused a ~267x slowdown (2500 it/s -> 9 it/s) on this dataset.
# A plain set-membership check on already-tokenized words is essentially
# free by comparison and catches the vast majority of real cases.
PROFANE_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "shit", "bitch", "bastard",
    "asshole", "dick", "pussy", "cunt", "cock", "whore", "slut", "nigger",
    "nigga", "faggot", "retard", "retarded", "twat", "wanker", "douche",
    "motherfucker", "bullshit", "dumbass", "jackass", "prick",
}

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())


def looks_like_gibberish(words) -> bool:
    """
    Heuristic check for text with almost no recognizable English words
    (spam / keyboard-mashing / joined-word junk).
    """
    if len(words) < MIN_WORD_COUNT:
        return True
    common_count = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    if common_count < 2:
        return True
    return False


def is_mostly_url(text: str) -> bool:
    """Rejects arguments that are just a link with barely any commentary."""
    stripped_of_urls = URL_PATTERN.sub("", text).strip()
    return len(stripped_of_urls) < MIN_ARG_CHARS


def passes_quality_filters(text: str) -> bool:
    if not (MIN_ARG_CHARS <= len(text) <= MAX_ARG_CHARS):
        return False
    if is_mostly_url(text):
        return False
    # Long run of letters with no space/punctuation = likely joined junk
    if re.search(r"[A-Za-z]{25,}", text):
        return False

    words = tokenize(text)
    if looks_like_gibberish(words):
        return False
    if any(w in PROFANE_WORDS for w in words):
        return False
    return True


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, strip weird control characters."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return text


def load_and_clean_arguments():
    """
    Stream-parses the locally downloaded args.me corpus JSON file
    (867MB — downloaded manually from https://zenodo.org/records/4139439)
    using ijson, so the whole file never has to sit in RAM at once.
    Filters out junk, dedupes, and stops once SAMPLE_SIZE good rows exist.

    The file's top-level shape is:
        { "arguments": [ { "id", "conclusion", "premises": [ {"text", "stance"} ] }, ... ] }
    """
    if not os.path.exists(RAW_JSON_PATH):
        raise FileNotFoundError(
            f"Could not find {RAW_JSON_PATH}\n"
            f"Download the corpus from https://zenodo.org/records/4139439, "
            f"unzip it, and place args-me-1.0-cleaned.json at that path."
        )

    print(f"Streaming local file: {RAW_JSON_PATH}")
    seen = set()
    rows = []

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        arguments_iter = ijson.items(f, "arguments.item")

        for arg in tqdm(arguments_iter, desc="Filtering arguments"):
            conclusion = clean_text(arg.get("conclusion", ""))
            arg_id = arg.get("id", "")
            premises = arg.get("premises", [])

            if not conclusion or not premises:
                continue

            # Each argument can have multiple premises (usually just one).
            for premise in premises:
                argument_text = clean_text(premise.get("text", ""))
                stance = premise.get("stance", "")

                if not argument_text:
                    continue
                if not passes_quality_filters(argument_text):
                    continue

                key = argument_text.lower()
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "topic": conclusion,
                    "argument": argument_text,
                    "stance": stance,
                    "source_title": "args.me corpus",
                    "source_url": f"https://www.args.me/api/v2/arguments/{arg_id}",
                })

            if len(rows) >= SAMPLE_SIZE:
                break

    print(f"Kept {len(rows)} cleaned arguments out of the file.")
    return pd.DataFrame(rows)


def build_faiss_index(df: pd.DataFrame, embedder: SentenceTransformer) -> faiss.Index:
    """
    Encodes every argument into an embedding and builds a FAISS index
    using cosine similarity (implemented as inner-product search on
    L2-normalized vectors — this is the standard, fast, accurate setup
    for sentence-transformer embeddings).
    """
    print("Encoding arguments into embeddings (this is the slow part)...")
    embeddings = embedder.encode(
        df["argument"].tolist(),
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_and_clean_arguments()
    if df.empty:
        raise RuntimeError("No arguments were collected — check your internet connection "
                            "or the dataset name on Hugging Face.")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    index = build_faiss_index(df, embedder)

    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving metadata to {METADATA_PATH}")
    df.to_csv(METADATA_PATH, index=False)

    print("\nDone! New dataset stats:")
    print(f"  Total arguments : {len(df)}")
    print(f"  Unique topics   : {df['topic'].nunique()}")
    print(f"  Stance counts   :\n{df['stance'].value_counts()}")


if __name__ == "__main__":
    main()