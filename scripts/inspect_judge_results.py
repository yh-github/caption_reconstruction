#!/usr/bin/env python3
"""
Interactive Pretty-Printer & Reviewer for QA Inspection Judge Results.

Displays manual and LLM-as-a-judge evaluation records with syntax highlighting,
boundary context, gap reconstructions, top-K candidates, and judge rationales.

Usage:
    # Step through records one-by-one interactively:
    ./.venv/bin/python scripts/inspect_judge_results.py

    # View a summary table of all records:
    ./.venv/bin/python scripts/inspect_judge_results.py --summary

    # Print all records continuously (e.g. to pipe to less):
    ./.venv/bin/python scripts/inspect_judge_results.py --all

    # Inspect a specific record by number (1-indexed):
    ./.venv/bin/python scripts/inspect_judge_results.py --index 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


DEFAULT_RESULTS_PATH = "results/qa_inspection/manual_judge_results_dev.jsonl"
DEFAULT_SOURCE_PATH = "results/qa_inspection/qa_inspection_dev_siglip.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def format_category_badge(category: Optional[str]) -> str:
    if not category:
        return "[dim]N/A[/dim]"
    cat = category.upper()
    if cat == "EXACT_FACT":
        return "[bold green]EXACT_FACT[/bold green]"
    elif cat == "SEMANTIC_TYPE":
        return "[bold cyan]SEMANTIC_TYPE[/bold cyan]"
    elif cat == "PLAUSIBLE_HALLUCINATION":
        return "[bold yellow]PLAUSIBLE_HALLUCINATION[/bold yellow]"
    elif cat == "MISSED":
        return "[bold red]MISSED[/bold red]"
    return f"[bold]{category}[/bold]"


def format_leakage_badge(leaked: Optional[bool]) -> str:
    if leaked is None:
        return "[dim]N/A[/dim]"
    if leaked:
        return "[bold red on #331111] ⚠ LEAKED [/bold red on #331111]"
    return "[bold green] CLEAN [/bold green]"


def format_answerable_badge(can_answer: Optional[bool]) -> str:
    if can_answer is None:
        return "[dim]N/A[/dim]"
    if can_answer:
        return "[bold green]YES[/bold green]"
    return "[bold red]NO (UNANSWERABLE)[/bold red]"


def find_matching_source(
    record: Dict[str, Any],
    index: int,
    source_records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Accurately match a judge record with its source QA record without key collisions."""
    if not source_records:
        return None
    # 1. Direct index check
    if 0 <= index < len(source_records):
        candidate = source_records[index]
        if candidate.get("question") == record.get("question") and candidate.get("assignment_id") == record.get("assignment_id"):
            return candidate

    # 2. Match on (assignment_id, question)
    for src in source_records:
        if src.get("assignment_id") == record.get("assignment_id") and src.get("question") == record.get("question"):
            return src

    # 3. Match on question text
    for src in source_records:
        if src.get("question") == record.get("question"):
            return src

    return None


def render_record_rich(
    console: Console,
    record: Dict[str, Any],
    source_record: Optional[Dict[str, Any]],
    index: int,
    total: int,
) -> None:
    assignment_id = record.get("assignment_id", "Unknown")
    question = record.get("question", "")
    gt_answer = record.get("ground_truth_answer", "")
    video_id = source_record.get("video_id", "Unknown") if source_record else "N/A"
    domain = source_record.get("domain", "") if source_record else ""
    evidence_span = source_record.get("evidence_span", "N/A") if source_record else "N/A"
    duration = source_record.get("evidence_duration_sec", "") if source_record else ""

    # Header Panel
    header_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    header_table.add_column("Key", style="bold cyan", width=18)
    header_table.add_column("Value", style="white")

    header_table.add_row("Video ID", f"[bold yellow]{video_id}[/bold yellow] {f'({domain})' if domain else ''}")
    header_table.add_row("Assignment ID", f"[dim]{assignment_id}[/dim]")
    header_table.add_row("Evidence Span", f"{evidence_span} {f'({duration}s gap)' if duration else ''}")
    header_table.add_row("Question", f"[bold white]{question}[/bold white]")
    header_table.add_row("Ground Truth", f"[bold green]{gt_answer}[/bold green]")

    title_text = f" RECORD {index + 1} of {total} "
    console.print(Panel(header_table, title=f"[bold white on blue]{title_text}[/bold white on blue]", border_style="blue", box=box.ROUNDED))

    # Context & Captions (from source record if available)
    if source_record:
        b_context = source_record.get("boundary_context", {})
        pre_gap = b_context.get("pre_gap", [])
        post_gap = b_context.get("post_gap", [])
        recon_captions = source_record.get("gap_conditions", {}).get("reconstructed") or []

        ctx_table = Table(box=box.SIMPLE, expand=True, padding=(0, 1))
        ctx_table.add_column("Section", style="bold magenta", width=22)
        ctx_table.add_column("Captions", style="white")

        # Pre-gap
        if pre_gap:
            pre_str = "\n".join(fr"[cyan]\[{c.get('timestamp')}][/cyan] {c.get('caption')}" for c in pre_gap)
        else:
            pre_str = "[dim](Video start - no pre-gap context)[/dim]"
        ctx_table.add_row("Pre-Gap Context", pre_str)

        # In-fill reconstructed
        if record.get("recon_available"):
            if recon_captions:
                recon_str = "\n".join(
                    fr"[yellow]\[{c.get('timestamp')}][/yellow] {c.get('caption') or '[dim]<empty>[/dim]'}"
                    for c in recon_captions
                )
            else:
                recon_str = "[dim](Reconstruction empty)[/dim]"
        else:
            recon_str = "[dim red](No reconstruction available for this gap)[/dim red]"
        ctx_table.add_row("Reconstructed In-Fill", recon_str)

        # Post-gap
        if post_gap:
            post_str = "\n".join(fr"[cyan]\[{c.get('timestamp')}][/cyan] {c.get('caption')}" for c in post_gap)
        else:
            post_str = "[dim](Video end - no post-gap context)[/dim]"
        ctx_table.add_row("Post-Gap Context", post_str)

        console.print(Panel(ctx_table, title="[bold]Surrounding Context & Gap In-Fill[/bold]", border_style="cyan", box=box.ROUNDED))

    # Task A: Intrinsic Recovery Verdict
    intrinsic = record.get("intrinsic_recovery")
    if intrinsic:
        t_a = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t_a.add_column("Field", style="bold", width=22)
        t_a.add_column("Value")

        leak_badge = format_leakage_badge(intrinsic.get("boundary_leakage"))
        leak_exp = intrinsic.get("leakage_explanation")
        t_a.add_row("Boundary Leakage", f"{leak_badge} {f'- [dim]{leak_exp}[/dim]' if leak_exp else ''}")

        cat_badge = format_category_badge(intrinsic.get("factual_recovery_category"))
        exp = intrinsic.get("explanation", "")
        t_a.add_row("Factual Recovery", f"{cat_badge}")
        t_a.add_row("Judge Rationale", f"[italic white]{exp}[/italic white]")

        console.print(Panel(t_a, title="[bold]Task A: Intrinsic Window Factual Recovery[/bold]", border_style="green", box=box.ROUNDED))
    else:
        console.print(Panel("[dim italic]No reconstruction evaluated for Task A.[/dim italic]", title="Task A: Intrinsic Recovery", border_style="dim"))

    # Task B: Downstream Retrieval & Answerability
    masked_qa = record.get("downstream_masked")
    recon_qa = record.get("downstream_reconstructed")

    qa_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    qa_table.add_column("Evaluation Dimension", style="bold", width=22)
    qa_table.add_column("Masked Condition (Evidence Removed)", style="white")
    qa_table.add_column("Reconstructed Condition (In-Filled)", style="white")

    # Top candidates if available
    if source_record and "top_k_candidates" in source_record:
        top_k = source_record["top_k_candidates"]
        m_cands = top_k.get("masked", [])
        r_cands = top_k.get("reconstructed", [])

        m_cands_str = "\n".join(
            fr"[dim]#{c.get('rank')}[/dim] [cyan]\[{c.get('timestamp')}][/cyan] {c.get('caption')}"
            for c in m_cands
        ) if m_cands else "[dim]None[/dim]"

        r_cands_str = "\n".join(
            fr"[dim]#{c.get('rank')}[/dim] [yellow]\[{c.get('timestamp')}][/yellow] {c.get('caption') or '[dim]<empty>[/dim]'}"
            for c in r_cands
        ) if r_cands else "[dim]N/A[/dim]"

        qa_table.add_row("Top-K Candidates", m_cands_str, r_cands_str)

    # Can Answer
    m_can = format_answerable_badge(masked_qa.get("can_answer") if masked_qa else None)
    r_can = format_answerable_badge(recon_qa.get("can_answer") if recon_qa else None)
    qa_table.add_row("Can Answer?", m_can, r_can)

    # Predicted Answer
    m_pred = masked_qa.get("predicted_answer", "N/A") if masked_qa else "N/A"
    r_pred = recon_qa.get("predicted_answer", "N/A") if recon_qa else "N/A"
    m_ts = fr" [dim]\[at {masked_qa.get('supporting_timestamp')}][/dim]" if masked_qa and masked_qa.get("supporting_timestamp") else ""
    r_ts = fr" [dim]\[at {recon_qa.get('supporting_timestamp')}][/dim]" if recon_qa and recon_qa.get("supporting_timestamp") else ""

    qa_table.add_row("Predicted Answer", f"[bold]{m_pred}[/bold]{m_ts}", f"[bold]{r_pred}[/bold]{r_ts}")

    console.print(Panel(qa_table, title="[bold]Task B: Downstream Retrieval Answerability & Necessity[/bold]", border_style="yellow", box=box.ROUNDED))
    console.print()


def render_record_plain(
    record: Dict[str, Any],
    source_record: Optional[Dict[str, Any]],
    index: int,
    total: int,
) -> None:
    separator = "=" * 80
    print(f"\n{separator}")
    print(f" RECORD {index + 1} / {total} | ID: {record.get('assignment_id')}")
    if source_record:
        print(f" Video: {source_record.get('video_id')} | Domain: {source_record.get('domain')}")
        print(f" Evidence Span: {source_record.get('evidence_span')} ({source_record.get('evidence_duration_sec')}s)")
    print(f" Question: {record.get('question')}")
    print(f" Ground Truth: {record.get('ground_truth_answer')}")
    print(separator)

    if source_record:
        b_context = source_record.get("boundary_context", {})
        print("\n--- BOUNDARY CONTEXT ---")
        print(" Pre-gap:")
        for c in b_context.get("pre_gap", []):
            print(f"   [{c.get('timestamp')}] {c.get('caption')}")
        print(" Post-gap:")
        for c in b_context.get("post_gap", []):
            print(f"   [{c.get('timestamp')}] {c.get('caption')}")

        print("\n--- RECONSTRUCTED IN-FILL ---")
        recon = source_record.get("gap_conditions", {}).get("reconstructed") or []
        for c in recon:
            print(f"   [{c.get('timestamp')}] {c.get('caption')}")

    intrinsic = record.get("intrinsic_recovery")
    print("\n--- TASK A: INTRINSIC RECOVERY ---")
    if intrinsic:
        print(f" Boundary Leakage: {intrinsic.get('boundary_leakage')} ({intrinsic.get('leakage_explanation')})")
        print(f" Factual Category: {intrinsic.get('factual_recovery_category')}")
        print(f" Judge Rationale : {intrinsic.get('explanation')}")
    else:
        print(" (No reconstruction available)")

    print("\n--- TASK B: DOWNSTREAM RETRIEVAL QA ---")
    m = record.get("downstream_masked")
    r = record.get("downstream_reconstructed")
    print(f" MASKED        -> Can Answer: {m.get('can_answer') if m else 'N/A'} | Ans: {m.get('predicted_answer') if m else 'N/A'} (at {m.get('supporting_timestamp') if m else 'N/A'})")
    print(f" RECONSTRUCTED -> Can Answer: {r.get('can_answer') if r else 'N/A'} | Ans: {r.get('predicted_answer') if r else 'N/A'} (at {r.get('supporting_timestamp') if r else 'N/A'})")
    print(f"{separator}\n")


def print_summary_table(
    console: Optional[Console],
    records: List[Dict[str, Any]],
    source_records: List[Dict[str, Any]],
) -> None:
    if console and HAS_RICH:
        table = Table(
            title="[bold blue]Summary of Judge Results[/bold blue]",
            box=box.ROUNDED,
            show_lines=True,
            expand=True,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Video / Question", style="white", min_width=30, max_width=45)
        table.add_column("Leakage?", justify="center", width=10)
        table.add_column("Factual Category", justify="center", width=24)
        table.add_column("Masked Ans?", justify="center", width=12)
        table.add_column("Recon Ans?", justify="center", width=12)

        for i, rec in enumerate(records):
            src = find_matching_source(rec, i, source_records) or {}
            vid = src.get("video_id", "Unknown")
            q = rec.get("question", "")
            gt = rec.get("ground_truth_answer", "")

            leak = rec.get("intrinsic_recovery", {}).get("boundary_leakage") if rec.get("intrinsic_recovery") else None
            leak_str = "[bold red]YES[/bold red]" if leak else ("[bold green]NO[/bold green]" if leak is False else "[dim]-[/dim]")

            cat = rec.get("intrinsic_recovery", {}).get("factual_recovery_category") if rec.get("intrinsic_recovery") else None
            cat_str = format_category_badge(cat)

            m_can = rec.get("downstream_masked", {}).get("can_answer") if rec.get("downstream_masked") else None
            m_str = "[bold green]YES[/bold green]" if m_can else ("[bold red]NO[/bold red]" if m_can is False else "[dim]-[/dim]")

            r_can = rec.get("downstream_reconstructed", {}).get("can_answer") if rec.get("downstream_reconstructed") else None
            r_str = "[bold green]YES[/bold green]" if r_can else ("[bold red]NO[/bold red]" if r_can is False else "[dim]-[/dim]")

            q_display = f"[bold yellow]{vid}[/bold yellow]\nQ: [white]{q}[/white]\n[dim]GT: {gt}[/dim]"
            table.add_row(str(i + 1), q_display, leak_str, cat_str, m_str, r_str)

        console.print(table)
    else:
        print(f"{'#':<4} | {'Video':<25} | {'Leakage':<8} | {'Category':<22} | {'Masked':<8} | {'Recon':<8}")
        print("-" * 80)
        for i, rec in enumerate(records):
            src = find_matching_source(rec, i, source_records) or {}
            vid = src.get("video_id", "Unknown")[:24]
            leak = str(rec.get("intrinsic_recovery", {}).get("boundary_leakage") if rec.get("intrinsic_recovery") else "-")
            cat = str(rec.get("intrinsic_recovery", {}).get("factual_recovery_category") if rec.get("intrinsic_recovery") else "-")
            m_can = str(rec.get("downstream_masked", {}).get("can_answer") if rec.get("downstream_masked") else "-")
            r_can = str(rec.get("downstream_reconstructed", {}).get("can_answer") if rec.get("downstream_reconstructed") else "-")
            print(f"{i + 1:<4} | {vid:<25} | {leak:<8} | {cat:<22} | {m_can:<8} | {r_can:<8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and review QA evaluation judge results one-by-one or in summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to judge results JSONL (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_PATH,
        help=f"Path to inspection source dataset JSONL (default: {DEFAULT_SOURCE_PATH})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a high-level summary table of all records and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all records continuously without interactive pausing.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Inspect a specific record number directly (1-indexed, e.g. --index 3).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )

    args = parser.parse_args()

    # Load data
    results = load_jsonl(args.results)
    if not results:
        print(f"Error: No records found in results file '{args.results}'.", file=sys.stderr)
        sys.exit(1)

    source_records = load_jsonl(args.source)

    console = Console() if HAS_RICH and not args.no_color else None

    # Summary table mode
    if args.summary:
        print_summary_table(console, results, source_records)
        return

    # Specific index mode
    if args.index is not None:
        idx = args.index - 1
        if 0 <= idx < len(results):
            rec = results[idx]
            src = find_matching_source(rec, idx, source_records)
            if console:
                render_record_rich(console, rec, src, idx, len(results))
            else:
                render_record_plain(rec, src, idx, len(results))
        else:
            print(f"Error: Index {args.index} out of range (1 to {len(results)}).", file=sys.stderr)
            sys.exit(1)
        return

    # All records mode (continuous)
    if args.all or not sys.stdin.isatty():
        for i, rec in enumerate(results):
            src = find_matching_source(rec, i, source_records)
            if console:
                render_record_rich(console, rec, src, i, len(results))
            else:
                render_record_plain(rec, src, i, len(results))
        return

    # Interactive Step-by-Step Mode (Default in TTY)
    curr_idx = 0
    total = len(results)

    while 0 <= curr_idx < total:
        rec = results[curr_idx]
        src = find_matching_source(rec, curr_idx, source_records)

        if console:
            render_record_rich(console, rec, src, curr_idx, total)
        else:
            render_record_plain(rec, src, curr_idx, total)

        prompt = f"[Record {curr_idx + 1}/{total}] Enter: next | b: back | 1-{total}: jump | q: quit > "
        try:
            user_input = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input in ("q", "quit", "exit"):
            break
        elif user_input in ("b", "back", "prev", "p"):
            curr_idx = max(0, curr_idx - 1)
        elif user_input.isdigit():
            target = int(user_input) - 1
            if 0 <= target < total:
                curr_idx = target
            else:
                print(f"Invalid record number. Must be between 1 and {total}.")
        else:
            # Enter or anything else proceeds to next
            curr_idx += 1
            if curr_idx >= total:
                print("\nReached the end of evaluated records.")
                break


if __name__ == "__main__":
    main()
