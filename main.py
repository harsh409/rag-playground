"""
Command-line entry point.

  python main.py upload <file.pdf> [more.pdf ...]   add PDFs to the knowledge base
  python main.py docs                               list uploaded PDFs
  python main.py ask "question"                     run the 3-agent team once
  python main.py chat                               ask questions in a loop
  python main.py demo                               upload sample PDFs + run example cases
  python main.py reset                              wipe the vector DB
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import rag

ICONS = {"researcher": "🔎 RESEARCHER", "analyst": "🧮 ANALYST", "writer": "✍️  WRITER"}

DEMO_CASES = [
    # needs: weekend highway tariff + VoltPlus discount + idle fee + GST + refund rule
    "Hi, I'm a VoltPlus member. Last Saturday I charged 42 kWh at your Mumbai-Pune "
    "highway ultra-fast (150 kW) station. My car stayed plugged in 35 minutes after "
    "charging finished. Before that, my first attempt failed after 3 minutes because the "
    "charger showed a fault, and I was still charged Rs 150. What should my total bill be, "
    "and can I get the Rs 150 back?",
    # needs: fleet tier discount + depot night rate + minimum commitment + SLA credit
    "We are a fleet partner. This month we used 1,600 kWh, all at our depot between "
    "midnight and 5 am. One of our depot chargers was down for 20 hours before it was "
    "fixed. How much is our energy invoice and do we get any SLA credit?",
]


def print_event(e: dict) -> None:
    name = ICONS.get(e["agent"], e["agent"])
    if e.get("status") == "start":
        rnd = f" (round {e['round']})" if e.get("round", 1) > 1 else ""
        rnd = " (revising after review)" if e.get("revision") else rnd
        print(f"\n{'─' * 70}\n{name} is working{rnd}...", flush=True)
        return
    for c in e.get("tools", []):
        print(f"   🔧 {c}")
    print(textwrap.indent(e["output"], "   │ "))


def run_question(q: str) -> None:
    from graph import ask

    if not rag.list_documents():
        sys.exit("Knowledge base is empty. Upload a PDF first:  python main.py upload file.pdf")
    print(f"\n❓ {q}")
    state = ask(q, on_event=print_event)
    print(f"\n{'═' * 70}\n✅ FINAL ANSWER\n{'═' * 70}\n{state['answer']}\n")


def upload(paths: list[str]) -> None:
    for p in paths:
        n = rag.ingest_pdf(p)
        print(f"📄 uploaded {Path(p).name}: {n} chunks indexed")


def main() -> None:
    ap = argparse.ArgumentParser(description="LangChain multi-agent RAG demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload").add_argument("pdfs", nargs="+")
    sub.add_parser("docs")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    sub.add_parser("demo").add_argument("--case", type=int, choices=[1, 2])
    sub.add_parser("reset")
    args = ap.parse_args()

    if args.cmd == "upload":
        upload(args.pdfs)
    elif args.cmd == "docs":
        docs = rag.list_documents()
        print("\n".join(f"  {k}: {v} chunks" for k, v in docs.items()) or "  (none)")
    elif args.cmd == "ask":
        run_question(args.question)
    elif args.cmd == "chat":
        while (q := input("\nYour question (blank to quit): ").strip()):
            run_question(q)
    elif args.cmd == "demo":
        from make_sample_pdfs import build_all

        upload([str(p) for p in build_all()])
        cases = [DEMO_CASES[args.case - 1]] if args.case else DEMO_CASES
        for q in cases:
            run_question(q)
    elif args.cmd == "reset":
        rag.reset()
        print("Vector DB wiped. Re-upload PDFs with: python main.py upload <file.pdf>")


if __name__ == "__main__":
    main()
