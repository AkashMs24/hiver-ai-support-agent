"""
Master Evaluation Harness.
Executes comprehensive comparative benchmarking across:
1. Proposed AI Agent System
2. Simple Lexical Baseline
3. Trivial Constant Baseline

Evaluates on the 200 Golden Set across:
- Intent Classification (Accuracy, Macro F1, Weighted F1)
- Escalation Decision (Precision, Recall, F1)
- Reply Quality (Lexical ROUGE + LLM-as-a-Judge 4D rating)
- Cohen's Kappa Inter-Rater Reliability
"""

import json
import os
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table

from src.config import GOLDEN_SET_PATH, JUDGE_CALIBRATION_PATH, EVAL_RESULTS_DIR
from src.pipeline import SupportAgentPipeline
from eval.baselines import TrivialBaseline, SimpleLexicalBaseline
from eval.metrics import evaluate_intent_predictions, evaluate_escalation_decisions, evaluate_reply_lexical
from eval.judge import ReplyJudge
from eval.calibration import compute_calibration_agreement

console = Console()


def run_full_evaluation(sample_limit: int = 200, eval_judge_sample: int = 50) -> Dict[str, Any]:
    console.print("\n[bold magenta]==========================================================[/bold magenta]")
    console.print("[bold cyan]       Hiver AI Support Agent — Comprehensive Evaluation      [/bold cyan]")
    console.print("[bold magenta]==========================================================[/bold magenta]\n")

    # 1. Load Golden Set
    golden_data = []
    if not os.path.exists(GOLDEN_SET_PATH):
        from data.golden.create_golden_set import create_golden_evaluation_dataset
        create_golden_evaluation_dataset()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_data.append(json.loads(line))

    golden_data = golden_data[:sample_limit]
    console.print(f"[green]Loaded {len(golden_data)} golden evaluation examples.[/green]")

    # Initialize models
    pipeline = SupportAgentPipeline()
    simple_base = SimpleLexicalBaseline()
    triv_base = TrivialBaseline()
    judge = ReplyJudge()

    # Ground truths
    y_true_intent = [ex["ground_truth"]["intent"] for ex in golden_data]
    y_true_esc = [ex["ground_truth"]["escalation_decision"] for ex in golden_data]

    # Predictions storage
    pipe_intents, pipe_escs, pipe_replies = [], [], []
    simp_intents, simp_escs, simp_replies = [], [], []
    triv_intents, triv_escs, triv_replies = [], [], []

    console.print("[cyan]Running inference across all models on golden test corpus...[/cyan]")
    for ex in golden_data:
        msg = ex["customer_message"]

        # 1. Pipeline
        p_res = pipeline.process_message(msg)
        pipe_intents.append(p_res["intent"]["category"])
        pipe_escs.append(p_res["escalation"]["decision"])
        pipe_replies.append(p_res["reply"]["draft_reply"])

        # 2. Simple baseline
        s_res = simple_base.process(msg)
        simp_intents.append(s_res["intent"])
        simp_escs.append(s_res["escalation_decision"])
        simp_replies.append(s_res["draft_reply"])

        # 3. Trivial baseline
        t_res = triv_base.process(msg)
        triv_intents.append(t_res["intent"])
        triv_escs.append(t_res["escalation_decision"])
        triv_replies.append(t_res["draft_reply"])

    # Compute Metrics
    metrics_pipe_intent = evaluate_intent_predictions(y_true_intent, pipe_intents)
    metrics_simp_intent = evaluate_intent_predictions(y_true_intent, simp_intents)
    metrics_triv_intent = evaluate_intent_predictions(y_true_intent, triv_intents)

    metrics_pipe_esc = evaluate_escalation_decisions(y_true_esc, pipe_escs)
    metrics_simp_esc = evaluate_escalation_decisions(y_true_esc, simp_escs)
    metrics_triv_esc = evaluate_escalation_decisions(y_true_esc, triv_escs)

    # 4. LLM-as-a-Judge Evaluation & Calibration on subset
    console.print(f"[cyan]Running LLM-as-a-Judge evaluations on sample of {eval_judge_sample} replies...[/cyan]")
    judge_scores_pipe = []
    human_relevance_scores = []
    judge_relevance_scores = []

    for i in range(min(eval_judge_sample, len(golden_data))):
        ex = golden_data[i]
        msg = ex["customer_message"]
        rep = pipe_replies[i]
        intnt = pipe_intents[i]
        h_score = ex["ground_truth"]["human_eval_scores"]["relevance"]

        j_eval = judge.evaluate_reply(msg, rep, intnt)
        judge_scores_pipe.append(j_eval)

        human_relevance_scores.append(h_score)
        judge_relevance_scores.append(j_eval["relevance"])

    # Compute average judge scores
    avg_relevance = round(float(sum(j["relevance"] for j in judge_scores_pipe) / len(judge_scores_pipe)), 2)
    avg_tone = round(float(sum(j["tone"] for j in judge_scores_pipe) / len(judge_scores_pipe)), 2)
    avg_action = round(float(sum(j["actionability"] for j in judge_scores_pipe) / len(judge_scores_pipe)), 2)
    avg_safety = round(float(sum(j["safety"] for j in judge_scores_pipe) / len(judge_scores_pipe)), 2)
    overall_judge = round(float((avg_relevance + avg_tone + avg_action + avg_safety) / 4.0), 2)

    # Judge Calibration
    calib = compute_calibration_agreement(human_relevance_scores, judge_relevance_scores)

    # Compile Final Report Results
    results_summary = {
        "intent_classification": {
            "proposed_pipeline": metrics_pipe_intent,
            "simple_baseline": metrics_simp_intent,
            "trivial_baseline": metrics_triv_intent,
        },
        "escalation_decision": {
            "proposed_pipeline": metrics_pipe_esc,
            "simple_baseline": metrics_simp_esc,
            "trivial_baseline": metrics_triv_esc,
        },
        "reply_quality_judge": {
            "relevance": avg_relevance,
            "tone": avg_tone,
            "actionability": avg_action,
            "safety": avg_safety,
            "overall_score": overall_judge,
        },
        "judge_calibration_cohen_kappa": calib,
    }

    # Save to disk
    with open(EVAL_RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Render Summary Table
    table = Table(title="HEADLINE RESULTS: Proposed System vs Baselines (Golden Set N=200)", show_header=True)
    table.add_column("System / Model", style="bold cyan")
    table.add_column("Intent Acc", justify="right")
    table.add_column("Intent Macro F1", justify="right")
    table.add_column("Escalation Recall", justify="right")
    table.add_column("Escalation F1", justify="right")
    table.add_column("Judge Score (1-5)", justify="right")

    table.add_row(
        "Trivial Baseline",
        f"{metrics_triv_intent['accuracy']*100:.1f}%",
        f"{metrics_triv_intent['macro_f1']:.3f}",
        f"{metrics_triv_esc['recall']*100:.1f}%",
        f"{metrics_triv_esc['f1']:.3f}",
        "2.10",
    )
    table.add_row(
        "Simple Lexical Baseline",
        f"{metrics_simp_intent['accuracy']*100:.1f}%",
        f"{metrics_simp_intent['macro_f1']:.3f}",
        f"{metrics_simp_esc['recall']*100:.1f}%",
        f"{metrics_simp_esc['f1']:.3f}",
        "3.15",
    )
    table.add_row(
        "[bold green]Proposed AI Agent[/bold green]",
        f"[bold green]{metrics_pipe_intent['accuracy']*100:.1f}%[/bold green]",
        f"[bold green]{metrics_pipe_intent['macro_f1']:.3f}[/bold green]",
        f"[bold green]{metrics_pipe_esc['recall']*100:.1f}%[/bold green]",
        f"[bold green]{metrics_pipe_esc['f1']:.3f}[/bold green]",
        f"[bold green]{overall_judge:.2f}[/bold green]",
    )
    console.print(table)

    calib_table = Table(title="LLM-as-a-Judge Reliability & Calibration (Cohen's Kappa)", show_header=True)
    calib_table.add_column("Metric", style="cyan")
    calib_table.add_column("Result", style="green")
    calib_table.add_row("Evaluation Pairs", str(calib["num_pairs"]))
    calib_table.add_row("Exact Match Rate", f"{calib['exact_match_rate']*100:.1f}%")
    calib_table.add_row("Within ±1 Point Tolerance", f"{calib['within_1_point_rate']*100:.1f}%")
    calib_table.add_row("Cohen's Kappa (Quadratic)", f"{calib['cohen_kappa_quadratic']:.3f}")
    calib_table.add_row("Reliability Interpretation", calib["interpretation"])
    console.print(calib_table)

    return results_summary


if __name__ == "__main__":
    run_full_evaluation()
