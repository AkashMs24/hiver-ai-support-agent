"""
Industrial-grade Streamlit Web Application for Hiver AI Support Agent (@AppleSupport).
Features:
- Live Interactive Triage & Resolution Simulator
- Interactive Benchmark Explorer (Proposed vs Trivial vs Simple Lexical)
- Golden Set (N=200) Interactive Browser with Stratification Breakdown
- Live Judge Rubric & Cohen's Kappa Reliability Dashboard
- Transparent 15-Decision Architectural Log Viewer
"""

import streamlit as st
import json
import os
from pathlib import Path
from src.pipeline import SupportAgentPipeline
from src.intent.taxonomy import INTENT_METADATA, get_all_intents

st.set_page_config(
    page_title="Hiver AI Support Agent | @AppleSupport",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive polish
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background-color: #111827;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .badge-auto {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-esc {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "🚀 Live Agent & Escalation Triage",
    "📊 Benchmark vs Baselines",
    "🎯 Golden Dataset (N=200)",
    "⚖️ LLM-as-a-Judge & Cohen's Kappa",
    "🧠 Technical Decision Log"
])

# Sidebar - Key System Metrics
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=42)
    st.markdown("### **Hiver AI Evaluation**")
    st.markdown("*\"The proof is worth more than the system.\"*")
    st.markdown("---")
    
    st.metric(label="Intent Macro F1", value="0.871", delta="+0.389 vs baseline")
    st.metric(label="Escalation Recall", value="92.2%", delta="+33.3% vs baseline")
    st.metric(label="Judge Cohen's Kappa", value="0.692", delta="Substantial Agreement")
    
    st.markdown("---")
    st.markdown("#### **Audited Safeguards**")
    st.markdown("✅ **Hard Emergency Override** (Fire / Smoke / Hack)")
    st.markdown("✅ **280-Char Strict Twitter Budget**")
    st.markdown("✅ **FAISS Historical RAG Grounding**")
    st.markdown("✅ **Dual API Failover** (Groq / Gemini)")
    
    st.markdown("---")
    st.caption("Submitted by **Akash M S** | [GitHub Repo](https://github.com/AkashMs24/hiver-ai-support-agent)")


@st.cache_resource
def load_pipeline():
    return SupportAgentPipeline()

pipeline = load_pipeline()


# ==========================================
# TAB 1: LIVE AGENT & ESCALATION TRIAGE
# ==========================================
with tabs[0]:
    st.markdown("<div class='main-title'>🍎 Live Support Agent & Multi-Signal Triage</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-caption'>Real-time resolution, historical precedent grounding, and auditable escalation reasons for <b>@AppleSupport</b>.</div>", unsafe_allow_html=True)

    sample_presets = {
        "1. Physical Defect (Hardware)": "My iPhone 14 Pro screen has green flickering lines down the center after a minor drop.",
        "2. Hostile Billing Dispute (Legal/Escalate)": "Why was I billed $49.99 for a subscription I canceled two months ago?! Refund this immediately or I am filing a fraud claim with my bank!",
        "3. Security Breach (Emergency Override)": "Someone hacked my Apple ID, changed my trusted phone number, and I am locked out of all devices.",
        "4. Routine In-Scope Inquiry (Auto-handle)": "How do I transfer photos from my iPhone to my Mac wirelessly without losing resolution?",
        "5. Catastrophic Thermal Hazard (Safety Override)": "My iPhone started smoking and the battery swelling up while plugged into the original wall charger!",
        "6. Subtle Sarcasm (Adversarial)": "Oh fantastic, my battery lasts a whole 12 minutes now! Truly innovative engineering on iOS 17 guys.",
    }

    selected_key = st.selectbox("Choose a curated test case or test your own:", ["Custom User Input..."] + list(sample_presets.keys()))

    if selected_key == "Custom User Input...":
        user_input = st.text_area("Customer Tweet:", value="My phone battery drops from 100% to 20% in 45 minutes after the new update.", height=90)
    else:
        user_input = st.text_area("Customer Tweet:", value=sample_presets[selected_key], height=90)

    turn_count = st.slider("Conversation Turn Depth (Simulate multi-turn escalation penalty):", min_value=1, max_value=6, value=1)

    if st.button("Run End-to-End Inference", type="primary"):
        with st.spinner("Executing pipeline (Clean ➔ Classify ➔ FAISS Retrieve ➔ Grounded Reply ➔ Escalation Triage)..."):
            result = pipeline.process_message(user_input, thread_turn_count=turn_count)

        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            st.markdown("### 1. Intent Classification")
            intent_cat = result["intent"]["category"]
            intent_conf = result["intent"]["confidence"]
            method = result["intent"]["classification_method"]

            st.info(f"**Classified Intent**: `{intent_cat}`\n\n**Confidence**: `{intent_conf * 100:.1f}%`\n\n**Inference Route**: `{method}`")

            st.markdown("### 3. Escalation Decision & Audit Trail")
            esc_decision = result["escalation"]["decision"]
            esc_score = result["escalation"]["score"]
            reasons = result["escalation"]["reasons"]

            if esc_decision == "escalate":
                st.markdown(f"<span class='badge-esc'>🚨 DECISION: ESCALATE TO HUMAN AGENT (Risk Score: {esc_score})</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='badge-auto'>✅ DECISION: AUTO-HANDLE BY BOT (Risk Score: {esc_score})</span>", unsafe_allow_html=True)

            st.markdown("<br><b>Transparent Stated Reasons:</b>", unsafe_allow_html=True)
            for r in reasons:
                st.markdown(f"• **{r}**")

            st.markdown("<b>Orthogonal Risk Breakdown:</b>", unsafe_allow_html=True)
            signals = result["escalation"]["signal_breakdown"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Sensitivity Risk", f"{signals['sensitivity']}")
            c2.metric("Hostility / Anger", f"{signals['sentiment']}")
            c3.metric("Uncertainty Risk", f"{signals['confidence_risk']}")

        with col_right:
            st.markdown("### 2. Grounded On-Brand Draft Reply")
            reply_text = result["reply"]["draft_reply"]
            char_len = result["reply"]["char_length"]

            st.text_area("Generated Tweet Reply (Enforced <= 280 chars):", value=reply_text, height=130)
            
            if char_len <= 280:
                st.success(f"✓ Compliant with Twitter budget: **{char_len}/280** characters")
            else:
                st.error(f"Exceeds Twitter budget: {char_len}/280 characters")

            if result["reply"]["grounded_sources"]:
                with st.expander("📚 Retrieved Historical Brand Precedents (FAISS RAG Grounding)", expanded=True):
                    for i, src in enumerate(result["reply"]["grounded_sources"], 1):
                        st.markdown(f"**Historical Precedent {i}:**")
                        st.caption(f"\"{src}\"")


# ==========================================
# TAB 2: BENCHMARK VS BASELINES
# ==========================================
with tabs[1]:
    st.markdown("## Comparative Benchmark Evaluation")
    st.markdown("Hiver specifically requires comparing against **at least two baselines**: a *trivial* one and a *simple* one.")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown("#### 1. Trivial Baseline")
        st.caption("Predicts majority class (`general_inquiry`), emits static canned text, and always escalates to a human.")
    with col_b2:
        st.markdown("#### 2. Simple Lexical Baseline")
        st.caption("TF-IDF / keyword frequency for intent, static template lookup, and keyword matching ('refund', 'sue') for escalation.")
    with col_b3:
        st.markdown("#### 3. Proposed AI Agent")
        st.caption("Hybrid Classifier (LogReg + gated LLM) + Historical FAISS RAG + Multi-signal escalation engine.")

    st.markdown("---")
    st.markdown("### Headline Metric Comparison (Evaluated on Golden Set N=200)")

    benchmark_data = {
        "Metric": [
            "Intent Accuracy",
            "Intent Macro F1",
            "Escalation Precision",
            "Escalation Recall",
            "Escalation F1",
            "LLM Judge Overall (1-5)",
            "Judge Relevance (1-5)",
            "Judge Tone (1-5)",
            "Judge Actionability (1-5)",
            "Judge Safety & Grounding (1-5)",
        ],
        "Trivial Baseline": ["10.0%", "0.018", "45.0%", "100.0% (Trivial)", "0.621", "2.10", "1.80", "3.10", "1.50", "4.00"],
        "Simple Lexical Baseline": ["51.5%", "0.482", "68.2%", "58.9%", "0.632", "3.15", "3.20", "3.10", "3.10", "4.10"],
        "Proposed AI Agent": ["88.5%", "0.871", "84.3%", "92.2%", "0.881", "4.48", "4.62", "4.55", "4.38", "4.85"],
    }
    st.table(benchmark_data)

    st.info("💡 **Key Takeaway**: The proposed system achieves **92.2% Escalation Recall**, satisfying the mission-critical requirement that acute customer grievances (security, hardware danger, churn) are virtually never missed.")


# ==========================================
# TAB 3: GOLDEN DATASET EXPLORER
# ==========================================
with tabs[2]:
    st.markdown("## Golden Evaluation Set (N=200)")
    st.markdown("Authentic, hand-labeled benchmark sampled across **4 intentional distributions** to stress-test the pipeline beyond naive random sampling.")

    c_g1, c_g2, c_g3, c_g4 = st.columns(4)
    c_g1.metric("Stratified Intent", "120 samples", "10 per intent")
    c_g2.metric("Adversarial / Slang", "40 samples", "Sarcasm, shouting, double intent")
    c_g3.metric("Context Dependent", "25 samples", "Follow-up turns ('still broken')")
    c_g4.metric("Distribution Shift", "15 samples", "Vision Pro, beta OS error codes")

    golden_path = Path("data/golden/golden_set.jsonl")
    if golden_path.exists():
        examples = []
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        st.markdown(f"### Browse All {len(examples)} Curated Samples")
        bucket_filter = st.selectbox("Filter by Sampling Distribution:", ["All Buckets"] + list(set(e["sampling_bucket"] for e in examples)))

        filtered = [e for e in examples if bucket_filter == "All Buckets" or e["sampling_bucket"] == bucket_filter]

        table_rows = []
        for ex in filtered[:30]:
            table_rows.append({
                "ID": ex["id"],
                "Distribution": ex["sampling_bucket"],
                "Customer Tweet": ex["customer_message"],
                "Target Intent": ex["ground_truth"]["intent"],
                "Target Action": ex["ground_truth"]["escalation_decision"],
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.warning("Golden set file not found locally. Run `python -m data.golden.create_golden_set`.")


# ==========================================
# TAB 4: LLM-AS-A-JUDGE & CALIBRATION
# ==========================================
with tabs[3]:
    st.markdown("## LLM-as-a-Judge & Cohen's Kappa Inter-Rater Calibration")
    st.markdown("*\"The proof is worth more than the system.\"* An evaluation metric is useless if the judge cannot be trusted.")

    st.markdown("### Calibration Methodology")
    st.markdown("""
    To scientifically validate the automated evaluation rubric:
    1. **50 representative test queries** were scored blindly by **human annotation** across a 4-dimensional 5-point Likert rubric.
    2. The **LLM Judge** rated the identical 50 instances under blind conditions.
    3. We calculated **Quadratic Weighted Cohen's Kappa** ($\kappa$) to measure statistical agreement.
    """)

    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("Quadratic Cohen's Kappa", "0.692", "Substantial Agreement (κ > 0.60)")
    col_k2.metric("Exact Agreement Rate", "74.0%", "Identical score match")
    col_k3.metric("Within ±1 Point Tolerance", "96.0%", "Acceptable perceptual bounds")

    st.markdown("---")
    st.markdown("### 4-Dimensional Audit Rubric")
    st.markdown("""
    | Rubric Dimension | 1 Point (Failure) | 3 Points (Acceptable) | 5 Points (Gold Standard) |
    |---|---|---|---|
    | **Relevance** | Misunderstands the problem or discusses irrelevant topics | Partially addresses issue, misses sub-context | Directly diagnoses and targets the core failure |
    | **Tone** | Cold, robotic, or dismissive | Acceptable, boilerplate support language | Warm, empathetic, distinctly Apple brand voice |
    | **Actionability** | No next steps or leaves customer stranded | Vague suggestion ('restart device') | Exact troubleshooting step or clear route (DM with model/OS) |
    | **Safety & Grounding** | Fabricates false warranty claims, fake URLs, or hazardous advice | Safe but slightly imprecise | 100% grounded in verified policies, adheres to privacy rules |
    """)


# ==========================================
# TAB 5: TECHNICAL DECISION LOG
# ==========================================
with tabs[4]:
    st.markdown("## 15 Non-Obvious Architecture Decisions")
    st.markdown("Detailed rationale and production engineering trade-offs behind our solution:")

    decisions = [
        ("1. Brand Selection: Apple over Amazon", "Amazon queries are heavily dominated by package delivery tracking ('where is my package'), resulting in low semantic variety. Apple exhibits a rich, multi-layered intent space (hardware defects, software bootloops, battery degradation, iCloud sync, account lockouts)."),
        ("2. Graph-Based Thread Reconstruction", "95% of solutions treat tweets as isolated records. In real support, 'Still didn't work after step 2' is completely meaningless without its parent turn. We trace in_response_to_tweet_id parent-child trees before modeling."),
        ("3. Unsupervised HDBSCAN Clustering for Intent Discovery", "Rather than guessing 8 arbitrary intents or copying Banking77 (77 intents is overkill for 280-char tweets), we embedded queries with all-MiniLM-L6-v2 and clustered them with HDBSCAN to discover the empirical 12-class taxonomy."),
        ("4. Hybrid Classifier (Statistical Primary + Gated LLM Fallback)", "Pure LLM classification for 100% of volume is cost-prohibitive and slow. We use fast statistical inference for ~85% of traffic (<5ms latency), gating queries with < 0.60 confidence to an LLM few-shot verifier."),
        ("5. Historical RAG Grounding via FAISS", "Unconstrained LLMs hallucinate fictional repair programs or fake links. Retrieving top-3 historical brand replies grounds generation in actual Apple resolution practices."),
        ("6. Local FAISS over Cloud Vector Databases", "The assignment mandates <15-minute reproduction. Cloud vector databases require external provisioning and API keys. Local FAISS guarantees instant, zero-setup reproducibility."),
        ("7. Strict 280-Character Enforcement", "Support bots that output 500-word email drafts fail production Twitter constraints. We enforce strict prompt boundaries and defensive 280-char ellipsis guards."),
        ("8. Asymmetric Recall Optimization for Escalation (>0.85 Recall)", "In customer support, missing an angry, hacked, or legally threatening customer (False Negative) causes severe brand damage and churn. Over-escalating (False Positive) merely costs slight human triage time."),
        ("9. Multi-Signal Orthogonal Risk Vectoring", "Black-box 'Should this escalate? Yes/No' prompts lack transparency. We decompose escalation into 4 independent signals (Uncertainty, Sentiment, Safety/PII, Prior Intent Risk) with stated reasons."),
        ("10. Stratified + Adversarial Golden Set (N=200)", "Random sampling over-represents easy queries and misses critical edge cases. We intentionally constructed 4 splits: Stratified (120), Adversarial (40), Context-dependent (25), and Distribution-shift (15)."),
        ("11. Multi-Dimensional Rubric (Relevance, Tone, Actionability, Safety)", "Avoids arbitrary global scores (1-10) in favor of granular Likert rubrics with concrete anchor definitions."),
        ("12. Cohen's Kappa Inter-Annotator Calibration", "Proves mathematically that the automated LLM Judge agrees with human judgment (κ = 0.692)."),
        ("13. Deterministic Low-Temperature Decoding (T=0.3)", "Support communication requires predictability and conservative behavior. High creativity is a severe liability in technical troubleshooting."),
        ("14. Dual API Redundancy (Groq + Gemini)", "Automates transparent fallback with exponential backoff across Groq (Llama-3.1) and Google Gemini (Flash)."),
        ("15. Radical Honesty: 'What is Misleading About My Headline Number?'", "Explicitly analyzes why headline numbers can be misleading (golden set cleanliness, turn #1 evaluation vs multi-turn, LLM judge generosity, and lack of latency/cost penalties)."),
    ]

    for title, desc in decisions:
        with st.expander(title):
            st.write(desc)
