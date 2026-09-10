"""
Interactive CLI demo for Hiver evaluation.
Lets reviewers test any real message and observe:
- Intent classification (with confidence and method)
- Grounded RAG reply generation (with character count & historical reference)
- Multi-signal escalation decision (with transparent stated reasons)
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from src.pipeline import SupportAgentPipeline

console = Console()


def run_interactive_demo():
    console.print("\n[bold magenta]==========================================================[/bold magenta]")
    console.print("[bold cyan]   Hiver AI Support Agent (@AppleSupport) — Interactive Demo   [/bold cyan]")
    console.print("[bold magenta]==========================================================[/bold magenta]\n")

    pipeline = SupportAgentPipeline()

    sample_prompts = [
        "My iPhone screen started glitching with green vertical lines after the iOS 17 update.",
        "Why the hell was I charged $49.99 on my credit card?! I canceled this subscription weeks ago, fix this now or I will dispute with my bank!",
        "Someone hacked my Apple ID and changed the recovery email address. I am completely locked out.",
        "How do I transfer photos from my iPhone to my Mac wirelessly?",
        "My phone suddenly started smoking and the battery expanded while plugged into the charger!!",
    ]

    console.print("[bold yellow]Choose a sample prompt or type your own:[/bold yellow]")
    for i, s in enumerate(sample_prompts, 1):
        console.print(f"  [green]{i}.[/green] \"{s}\"")
    console.print("  [green]0.[/green] Custom manual input\n")

    while True:
        choice = Prompt.ask("Select option (1-5, 0 for custom, 'q' to quit)", default="1")
        if choice.lower() == "q":
            break

        if choice in ["1", "2", "3", "4", "5"]:
            msg = sample_prompts[int(choice) - 1]
        else:
            msg = Prompt.ask("\nEnter customer tweet")

        console.print(f"\n[cyan]Processing message:[/cyan] [italic]\"{msg}\"[/italic]...\n")
        res = pipeline.process_message(msg)

        # 1. Intent Table
        t_intent = Table(title="1. Intent Classification", show_header=True, header_style="bold cyan")
        t_intent.add_column("Property", style="dim")
        t_intent.add_column("Value", style="bold green")
        t_intent.add_row("Detected Intent", res["intent"]["category"])
        t_intent.add_row("Confidence", f"{res['intent']['confidence'] * 100:.1f}%")
        t_intent.add_row("Method Used", res["intent"]["classification_method"])
        console.print(t_intent)

        # 2. Reply Panel
        rep_panel = Panel(
            f"[bold white]{res['reply']['draft_reply']}[/bold white]\n\n"
            f"[dim]Length: {res['reply']['char_length']}/280 chars | Grounded in {len(res['reply']['grounded_sources'])} historical precedent(s)[/dim]",
            title="2. Draft Reply (Twitter Constrained)",
            border_style="blue",
        )
        console.print(rep_panel)

        # 3. Escalation Panel
        esc = res["escalation"]
        esc_color = "red" if esc["decision"] == "escalate" else "green"
        reasons_list = "\n".join([f"• {r}" for r in esc["reasons"]])

        esc_panel = Panel(
            f"Decision: [bold {esc_color}]{esc['decision'].upper()}[/bold {esc_color}] (Score: {esc['score']})\n\n"
            f"[bold]Stated Reasons:[/bold]\n{reasons_list}\n\n"
            f"[dim]Signals: Sensitivity={esc['signal_breakdown']['sensitivity']}, "
            f"Sentiment={esc['signal_breakdown']['sentiment']}, "
            f"ConfidenceRisk={esc['signal_breakdown']['confidence_risk']}, "
            f"PriorRisk={esc['signal_breakdown']['intent_prior']}[/dim]",
            title="3. Escalation Decision & Transparency",
            border_style=esc_color,
        )
        console.print(esc_panel)
        console.print("-" * 60)


if __name__ == "__main__":
    run_interactive_demo()
