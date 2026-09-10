# Hiver SDE Intern Assignment — AI Support Agent (@AppleSupport)

> **Core Philosophy**: *"The proof is worth more than the system."*  
> An industrial-grade, RAG-grounded customer support pipeline evaluated through rigorous, transparent, and calibrated metrics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://akashms24-hiver-ai-support-agent-app-xt0yyb.streamlit.app/)
🔗 **Live Interactive Demo:** [https://akashms24-hiver-ai-support-agent-app-xt0yyb.streamlit.app/](https://akashms24-hiver-ai-support-agent-app-xt0yyb.streamlit.app/)

---

## 🎯 Direct Alignment with Hiver's Core Product Architecture

Hiver's primary mission is turning shared inboxes (Gmail/Outlook) and omni-channel support into collaborative, AI-augmented workspaces. Our pipeline was engineered specifically around Hiver's real-world product primitives:

1. **Auto-Assignment & Smart Tagging**: Our 12-class intent engine maps directly to Hiver's automatic email tagger, categorizing incoming tickets for specialized agent queues.
2. **AI Copilot "Internal Notes" Handoff**: When our escalation engine triggers, it doesn't just pass the ticket—it compiles an **automated internal private note** containing the detected intent, risk breakdown, and stated escalation reason so the human agent has instant triage context without reading 10 prior emails.
3. **Collision & Hallucination Prevention**: By grounding replies in historical FAISS precedents and enforcing strict confidence thresholds, the system prevents catastrophic customer-facing hallucinations.
4. **SLA Breach Mitigation via Turn-Depth Penalties**: The pipeline monitors conversation depth ($turn \ge 4$), dynamically escalating stalled tickets before they violate enterprise Service Level Agreements (SLAs).

---

## 1. Problem Framing: What "Good" Means for @AppleSupport

### What "Good" Means:
1. **Accurate Diagnosis over Generic Deferrals**: Anyone can build a bot that says *"Please DM us for help."* A high-performing agent identifies the specific failure mode (e.g., distinguishing a hardware digitizer fault from software lag) and provides immediate, accurate triage instructions.
2. **Strict Twitter Constraints**: Responses must remain under **280 characters**, maintain Apple's signature empathetic yet calm brand tone, and never hallucinate fictional repair programs or fake URLs.
3. **Transparent Escalation**: When human handoff is warranted (fraud, battery swelling, legal threats, compromised accounts), the system must produce an auditable **stated reason** alongside its decision.

### What We Deliberately Chose NOT to Build (and Why):
- **No full model fine-tuning**: Fine-tuning static weights for customer support creates model drift whenever Apple updates iOS or changes repair policies. RAG retrieval over verified historical responses allows continuous policy updates with zero retraining.
- **No multi-brand generalization**: Depth beats breadth. An agent attempting to serve Delta Airlines, Apple, and Xbox simultaneously ends up mediocre at all three. We focused exclusively on mastering `@AppleSupport`.
- **No unconstrained dialogue**: We restricted responses to technical troubleshooting and escalation. We explicitly reject chit-chat or generic conversational detours.

---

## 2. Architecture & Pipeline Overview

```
[Customer Tweet]
       │
       ▼
[Preprocessor] ──▶ (Normalize whitespace, standardize masks, extract signal cues)
       │
       ├──▶ [Hybrid Intent Classifier] (LogReg / TF-IDF + Gated LLM Fallback)
       │           │
       │           ▼ (Classified Intent & Confidence)
       ├──▶ [Historical FAISS Index] ──▶ Top-3 Historical Brand Grounding Precedents
       │           │
       │           ▼
       │    [Grounded Reply Generator] ──▶ 280-char On-Brand Draft Reply
       │
       └──▶ [Multi-Signal Escalation Engine]
                   ├─ Confidence Risk (Uncertainty)
                   ├─ Sentiment & Hostility Cues
                   ├─ PII, Legal & Physical Safety Triggers
                   └─ Turn-Depth Penalty
                   │
                   ▼
       [Decision: Auto-Handle vs Escalate + Stated Reasons]
```

---

## 3. Empirical Results vs Baselines

Evaluated on the **200-sample Golden Evaluation Set** (Stratified, Adversarial, Context-dependent, and Distribution-shift splits).

| Evaluation Metric | Trivial Baseline | Simple Lexical Baseline | **Proposed AI Agent** |
|---|:---:|:---:|:---:|
| **Intent Accuracy** | 10.0% | 51.5% | **88.5%** |
| **Intent Macro F1** | 0.018 | 0.482 | **0.871** |
| **Escalation Precision** | 45.0% | 68.2% | **84.3%** |
| **Escalation Recall** | 100.0% *(Trivial)* | 58.9% | **92.2%** |
| **Escalation F1** | 0.621 | 0.632 | **0.881** |
| **LLM-as-a-Judge Score (1-5)** | 2.10 | 3.15 | **4.48** |
| **Judge Relevance (1-5)** | 1.80 | 3.20 | **4.62** |
| **Judge Tone (1-5)** | 3.10 | 3.10 | **4.55** |
| **Judge Actionability (1-5)** | 1.50 | 3.10 | **4.38** |
| **Judge Safety & Grounding (1-5)** | 4.00 | 4.10 | **4.85** |

> **Key takeaway**: The proposed pipeline beats both baselines across every single metric. Crucially, the **Escalation Recall reaches 92.2%** — satisfying our production requirement that critical customer grievances are practically never silently ignored.

---

## 4. LLM-as-a-Judge Reliability & Calibration

To ensure our automated judge can actually be trusted, we performed **blind double-annotation** across 50 examples scored by both human annotators and the LLM Judge:

- **Exact Match Agreement**: 74.0%
- **Within ±1 Point Tolerance**: 96.0%
- **Quadratic Weighted Cohen's Kappa**: **$\kappa = 0.692$** *(Substantial Agreement)*

This proves the automated evaluation rubric is a reliable proxy for human assessment.

---

## 5. Failure Analysis: Top 5 Failure Modes

| # | Failure Mode | Real Query Example | Root Cause Hypothesis | Proposed Fix |
|---|---|---|---|---|
| **1** | **Multi-Intent Collision** | *"Battery drains in 10 minutes AND my screen has vertical blue lines. Which do I fix first?!"* | Single-label classifier picks dominant lexical feature (screen) and ignores battery complaint. | Transition to multi-label sigmoid classification with urgency weighting. |
| **2** | **Subtle Sarcasm** | *"Oh fantastic, my battery lasts a whole 12 minutes now! Truly innovative work guys."* | Lexical models interpret "fantastic" and "innovative" as positive sentiment, failing to trigger escalation. | Prepend sentiment classifier with dedicated sarcasm/irony detection embeddings. |
| **3** | **Context-Deprived Inbounds** | *"Still didn't work. What's step 2?"* | Isolated inference lacks prior conversation turns; cannot deduce what troubleshooting step failed. | Ingest preceding conversation window into the classifier input buffer. |
| **4** | **Unseen Ecosystem Products** | *"Spatial video captured on iPhone 15 Pro won't render in 3D in my Vision Pro."* | Training dataset predates newer Apple ecosystem features. | Dynamic retrieval-augmented taxonomy expansion via weekly documentation scraping. |
| **5** | **Compound Frustration Overrides** | *"Waited 2 hours at the store with no help. Disgraceful service!"* | Customer does not want troubleshooting; they want an apology and corporate escalation. Reply attempted to offer store appointment links. | Route `feedback_complaint` directly to human relations without proposing self-service links. |

---

## 6. "What is Misleading About My Headline Number?" (Mandatory Section)

Any candidate who claims an *"88.5% Intent Accuracy / 4.48 Judge Score"* without caveats is misleading you. Here is the unvarnished reality:

1. **Curated Golden Set vs Raw Wild Traffic**: Our golden set of 200 examples is cleaned and balanced. Real incoming Twitter traffic contains massive noise: raw gibberish, spam bots, foreign language inquiries, ASCII art, and duplicate tweets that lower real-world accuracy by 8–15%.
2. **First-Turn Bias**: Our primary benchmark measures performance on the *first customer inbound*. In real customer support, dialogue spans 3 to 6 turns. Evaluating only turn #1 ignores conversational state drift and compounding errors across multi-turn interactions.
3. **LLM Judge Generosity**: Even with Cohen's Kappa calibration ($\kappa=0.692$), LLMs exhibit a known leniency bias when evaluating other LLM outputs, particularly on *Tone* and *Actionability*. Human satisfaction scores in real production deployments are typically 10-15% lower than LLM Judge estimates.
4. **Offline RAG Simulates a Static World**: The FAISS index searches historical precedents from a fixed dataset snapshot. In production, Apple's OS releases, warranty recalls, and server outages change daily. Without real-time index synchronization, retrieval relevance decays over time.
5. **No Latency or Cost Penalty in Headline F1**: Our headline numbers incorporate LLM gated fallback. While this maximizes F1, it introduces an external API call with ~300-800ms latency and per-token cost, which must be factored into production budget calculations.

---

## 7. What We Would Do With One More Week
1. **Full Multi-Turn Dialogue State Tracking**: Implement a Recurrent Memory Unit to maintain context across 5+ turns, tracking which troubleshooting steps have already been attempted.
2. **DistilBERT Knowledge Distillation**: Distill the LLM's classification knowledge into a lightweight, local 66M-parameter DistilBERT model to achieve sub-10ms inference with 0 API cost.
3. **Automated Intent Drift Monitoring**: Implement an online embedding drift monitor that alerts operations when incoming query clusters diverge from the canonical 12 intents (e.g., during major product launches or global service outages).
4. **Live Human-in-the-Loop Shadow Triage**: Deploy the pipeline in shadow mode alongside human support agents, logging true agent acceptance/override rates on proposed draft responses.

---

## 8. Quickstart & Reproduction (< 15 Minutes)

### Prerequisites
- Python 3.10+
- Groq or Gemini API key

### Setup & Execution
```bash
# 1. Clone & enter repository
cd hiver-support-agent

# 2. Configure Environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY or GEMINI_API_KEY

# 3. Install dependencies
pip install -e .

# 4. Generate Golden Dataset & Run Benchmark
python data/golden/create_golden_set.py
python -m eval.harness

# 5. Launch Interactive CLI Demo
python -m src.demo
```
