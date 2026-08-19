import streamlit as st
import pandas as pd
import rag_utils

st.set_page_config(page_title="Open-Domain Debater Assistant", page_icon="🧠")

st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Open-Domain Debater Assistant")
st.write("""
This app lets you input a debate topic, retrieves relevant arguments,
classifies them, and generates structured summaries (Point/Counterpoint).
""")


@st.cache_resource
def get_data():
    return rag_utils.load_data()


@st.cache_resource
def get_models():
    return rag_utils.load_models()


# ----------------------------------------------------------------------
# Load data + models, with friendly errors instead of a raw crash
# ----------------------------------------------------------------------
try:
    metadata, index = get_data()
except rag_utils.DataNotFoundError as e:
    st.error(f"⚠️ Dataset not found.\n\n{e}")
    st.stop()

try:
    embedder, classifier, summarizer = get_models()
except rag_utils.ModelLoadError as e:
    st.error(f"⚠️ Could not load AI models.\n\n{e}")
    st.stop()


# ----------------------------------------------------------------------
# Main interface
# ----------------------------------------------------------------------
query = st.text_input("Enter a debate topic:", placeholder="E.g. Should we ban plastic bags?")

if query:
    st.write(f"**Searching for arguments related to: {query}**")

    try:
        results = rag_utils.retrieve_arguments(embedder, index, metadata, query)
    except Exception as e:
        st.error(f"Something went wrong while searching: {e}")
        st.stop()

    if not results:
        st.warning("No results found. Try a different topic.")
        st.stop()

    best_similarity = max(r["similarity"] for r in results)
    if best_similarity < rag_utils.LOW_RELEVANCE_THRESHOLD:
        st.warning(
            "⚠️ We couldn't find strongly related arguments for this topic in our "
            "dataset. The results below are the closest matches we have, but they "
            "may not be very relevant — try rephrasing, or try a more common "
            "debate topic."
        )

    pro_arguments = []
    con_arguments = []
    chart_rows = []  # for the visual score chart, shown after the argument list

    for r in results:
        label, confidence = rag_utils.classify_argument(classifier, r["argument"])
        stance_score = rag_utils.calculate_stance_strength(r["similarity"], confidence)

        if r["stance"] == "PRO":
            pro_arguments.append(r["argument"])
            stance_badge = "👍 PRO"
        elif r["stance"] == "CON":
            con_arguments.append(r["argument"])
            stance_badge = "👎 CON"
        else:
            stance_badge = "❔ UNKNOWN"

        st.write(f"🟩 **Topic:** {r['topic']}")
        st.write(f"🗣️ **Argument:** {r['argument']}")
        st.write(f"⚖️ **Stance:** {stance_badge}")
        st.write(f"🏷️ **Type:** {label} (Confidence: {confidence})")
        st.write(f"📈 **Stance Strength Score:** {stance_score}")
        st.write(f"📚 **Source:** {r['source']}")
        st.write("---")

        # Keep a short label (long topics make the chart unreadable)
        short_topic = r["topic"] if len(r["topic"]) <= 40 else r["topic"][:37] + "..."
        chart_rows.append({
            "Result": f"{short_topic} ({r['stance'] or '?'})",
            "Similarity": round(r["similarity"], 3),
            "Stance Strength Score": stance_score,
        })

    st.markdown("### 📊 Retrieval Score Chart")
    st.caption(
        "Higher bars = the model found a stronger match to your topic. "
        "If all bars are short, the results above are probably weak matches."
    )
    chart_df = pd.DataFrame(chart_rows).set_index("Result")
    st.bar_chart(chart_df)

    st.write("🧠 Generating Point / Counterpoint summary using BART...")

    col1, col2 = st.columns(2)
    pro_summary = ""
    con_summary = ""

    with col1:
        st.markdown("### 👍 Supporting Arguments (PRO)")
        if pro_arguments:
            pro_summary = rag_utils.generate_structured_summary(summarizer, query, " ".join(pro_arguments))
            st.write(pro_summary)
        else:
            st.write("_No PRO arguments were found in the top results for this topic._")

    with col2:
        st.markdown("### 👎 Opposing Arguments (CON)")
        if con_arguments:
            con_summary = rag_utils.generate_structured_summary(summarizer, query, " ".join(con_arguments))
            st.write(con_summary)
        else:
            st.write("_No CON arguments were found in the top results for this topic._")

    structured_summary = " ".join(filter(None, [pro_summary, con_summary]))

    # ------------------------------------------------------------------
    # Optional evaluation
    # ------------------------------------------------------------------
    reference_summary = st.text_area("Enter a reference summary for evaluation (optional):")

    if reference_summary and structured_summary:
        try:
            from rouge_score import rouge_scorer
            from nltk.translate.bleu_score import sentence_bleu

            st.write("🔍 Evaluating summary quality with ROUGE and BLEU:")

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference_summary, structured_summary)
            st.write(f"**ROUGE Scores:** {rouge_scores}")

            bleu_score = sentence_bleu(
                [reference_summary.split()], structured_summary.split()
            )
            st.write(f"**BLEU Score:** {bleu_score}")
        except Exception as e:
            st.error(f"Could not compute evaluation scores: {e}")