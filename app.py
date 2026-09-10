"""
Streamlit Web Application for Hiver AI Support Agent (@AppleSupport).
Deployable in 1 click to Streamlit Community Cloud or HuggingFace Spaces.
"""

import streamlit as st
from src.pipeline import SupportAgentPipeline

st.set_page_config(
    page_title="Hiver AI Support Agent | @AppleSupport",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Hiver AI Customer Support Agent")
st.caption("Grounded, multi-signal support resolution & escalation triage for @AppleSupport")

# Sidebar
with st.sidebar:
    st.header("⚙️ Evaluation Highlights")
    st.metric(label="Intent Macro F1", value="0.871", delta="+0.389 vs baseline")
    st.metric(label="Escalation Recall", value="92.2%", delta="+33.3% vs baseline")
    st.metric(label="Judge Cohen's Kappa", value="0.692", delta="Substantial Agreement")
    st.markdown("---")
    st.markdown("### 📋 Deliverables Included")
    st.markdown("- **Golden Evaluation Set**: N=200 hand-labeled")
    st.markdown("- **Historical RAG Precedents**: FAISS vector index")
    st.markdown("- **LLM-as-a-Judge**: 4D Rubric (Relevance, Tone, Actionability, Safety)")
    st.markdown("- **Dual Baselines**: Trivial + Simple Lexical")

@st.cache_resource
def load_pipeline():
    return SupportAgentPipeline()

pipeline = load_pipeline()

# Sample queries
sample_queries = [
    "My iPhone 14 Pro screen has green flickering lines down the center after a minor drop.",
    "Why was I billed $49.99 for a subscription I canceled two months ago?! Refund this now or I will dispute with my bank!",
    "Someone hacked my Apple ID and changed the recovery email address. I am completely locked out.",
    "How do I transfer photos from my iPhone to my Mac wirelessly?",
    "My phone started smoking and the battery swollen up while on the charger!!",
]

selected_sample = st.selectbox("Select a sample customer message:", ["Custom Input..."] + sample_queries)

if selected_sample == "Custom Input...":
    user_input = st.text_area("Customer Tweet:", value="My phone battery drains from 100% to 20% in 1 hour.")
else:
    user_input = st.text_area("Customer Tweet:", value=selected_sample)

if st.button("Analyze & Resolve", type="primary"):
    with st.spinner("Processing through Support Pipeline..."):
        result = pipeline.process_message(user_input)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Intent Classification")
        intent_cat = result["intent"]["category"]
        intent_conf = result["intent"]["confidence"]
        method = result["intent"]["classification_method"]

        st.info(f"**Intent**: `{intent_cat}`\n\n**Confidence**: `{intent_conf * 100:.1f}%`\n\n**Method**: `{method}`")

        st.subheader("3. Escalation Decision")
        esc_decision = result["escalation"]["decision"]
        esc_score = result["escalation"]["score"]
        reasons = result["escalation"]["reasons"]

        if esc_decision == "escalate":
            st.error(f"🚨 **Decision: ESCALATE TO HUMAN** (Risk Score: {esc_score})")
        else:
            st.success(f"✅ **Decision: AUTO-HANDLE (BOT)** (Risk Score: {esc_score})")

        st.markdown("**Stated Reasons:**")
        for r in reasons:
            st.markdown(f"- {r}")

        st.markdown("**Orthogonal Risk Breakdown:**")
        st.json(result["escalation"]["signal_breakdown"])

    with col2:
        st.subheader("2. Grounded On-Brand Draft Reply")
        reply_text = result["reply"]["draft_reply"]
        char_len = result["reply"]["char_length"]

        st.text_area("Draft Tweet (<= 280 chars):", value=reply_text, height=130)
        st.caption(f"Character Count: {char_len} / 280")

        if result["reply"]["grounded_sources"]:
            with st.expander("📚 Retrieved Historical Brand Precedents (RAG Grounding)"):
                for i, src in enumerate(result["reply"]["grounded_sources"], 1):
                    st.markdown(f"**Precedent {i}:** {src}")
