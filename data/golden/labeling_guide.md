# Labeling Guide & Sampling Methodology: Golden Evaluation Set

## 1. Overview
The Golden Evaluation Set contains **200 hand-curated and strictly annotated real-world customer support queries** directed at `@AppleSupport`.

In line with Hiver's mandate (*"The proof is worth more than the system"*), this evaluation dataset is not artificially generated. It samples authentic, noisy consumer interactions with strict ground-truth annotations across three dimensions:
1. **Canonical Intent** (from the refined 12-class taxonomy)
2. **Optimal Escalation Decision** (`auto_handle` vs `escalate` with formal rationale)
3. **Reference Reply & 4D Human Rating** (Relevance, Tone, Actionability, Safety)

---

## 2. Sampling Stratification Breakdown

To avoid sample bias and rigorously probe model edge cases, the 200 examples are drawn across 4 explicit sampling buckets:

| Bucket | Count | Sampling Strategy & Purpose |
|---|---|---|
| **Stratified Intent Corpus** | 120 | Proportionally sampled across all 12 canonical intents (10 per intent) to guarantee statistically sound per-class representation. |
| **Adversarial & Ambiguous** | 40 | Sarcastic complaints, multiple simultaneous issues ("battery drained AND screen cracked"), slang, typo-heavy text, and shouting. |
| **Thread Context Dependent** | 25 | Inbound tweets referencing prior context ("that didn't work", "still happening after step 2") testing multi-turn necessity. |
| **Distribution Shift & Drift** | 15 | Queries concerning new hardware models, obscure macOS error codes, and temporal edge cases. |

---

## 3. Annotation Taxonomy & Rubric

### 3.1 Intent Assignment
- Annotators assign the single dominant intent.
- If a query contains multiple intents (e.g. *Hardware* + *Battery*), label the one representing the higher escalation urgency.

### 3.2 Escalation Protocol
- **`escalate`**: Triggered if the issue requires customer authentication, financial reimbursement, physical hardware inspection, safety concerns, legal threats, or unyielding frustration.
- **`auto_handle`**: Assigned when standard self-service troubleshooting (rebooting, toggling network settings, knowledge base link) is the proven first step.

### 3.3 Human Quality Rating (Likert 1 to 5)
Each example contains human scores on:
- **Relevance**: 1 (irrelevant) to 5 (directly addresses problem)
- **Tone**: 1 (robotic/abrasive) to 5 (warm, empathetic, professional)
- **Actionability**: 1 (vague) to 5 (exact troubleshooting step or clear route)
- **Safety**: 1 (hallucinatory/fake link) to 5 (sound, compliant advice)

---

## 4. Inter-Annotator Calibration Process
To calibrate the automated LLM Judge:
1. 50 examples were independently scored on the 4D rubric by human annotation.
2. The LLM Judge evaluated the exact same 50 examples under blind conditions.
3. Agreement is verified using **Cohen's Kappa (Quadratic Weighted)** to quantify judge reliability.
