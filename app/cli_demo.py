"""
cli_demo.py
------------
A terminal (no-browser) version of the Debater Assistant, useful for quick
testing without launching Streamlit. Uses the exact same logic as app.py
via rag_utils.py — this file replaces the old embed_and_index.py and
retrieve.py, which were two separate near-duplicates of this same script.

Run:
    python app/cli_demo.py
"""

import rag_utils


def main():
    print("Loading dataset...")
    try:
        metadata, index = rag_utils.load_data()
    except rag_utils.DataNotFoundError as e:
        print(f"\n❌ {e}")
        return

    print("Loading AI models (this can take a while on first run)...")
    try:
        embedder, classifier, summarizer = rag_utils.load_models()
    except rag_utils.ModelLoadError as e:
        print(f"\n❌ {e}")
        return

    print("\nReady! Type a debate topic, or 'exit' to quit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = rag_utils.retrieve_arguments(embedder, index, metadata, query)
        if not results:
            print("No results found.\n")
            continue

        best_similarity = max(r["similarity"] for r in results)
        if best_similarity < rag_utils.LOW_RELEVANCE_THRESHOLD:
            print("⚠️  No strongly related arguments found — showing closest matches anyway.\n")

        pro_arguments, con_arguments = [], []

        for r in results:
            label, confidence = rag_utils.classify_argument(classifier, r["argument"])
            stance_score = rag_utils.calculate_stance_strength(r["similarity"], confidence)

            if r["stance"] == "PRO":
                pro_arguments.append(r["argument"])
            elif r["stance"] == "CON":
                con_arguments.append(r["argument"])

            print(f"Topic: {r['topic']}")
            print(f"Argument: {r['argument']}")
            print(f"Stance: {r['stance']}")
            print(f"Type: {label} (Confidence: {confidence})")
            print(f"Stance Strength Score: {stance_score}")
            print(f"Source: {r['source']}")
            print("-" * 60)

        print("\nGenerating PRO/CON summaries...\n")

        if pro_arguments:
            pro_summary = rag_utils.generate_structured_summary(summarizer, query, " ".join(pro_arguments))
            print(f"PRO summary: {pro_summary}\n")
        else:
            print("PRO summary: (no PRO arguments found)\n")

        if con_arguments:
            con_summary = rag_utils.generate_structured_summary(summarizer, query, " ".join(con_arguments))
            print(f"CON summary: {con_summary}\n")
        else:
            print("CON summary: (no CON arguments found)\n")


if __name__ == "__main__":
    main()