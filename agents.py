"""
The 3 agents. Each one is a LangChain agent = LLM + system prompt + tools.

  Researcher  tools: search_documents (RAG)  -> gathers cited facts from the PDFs
  Analyst     tools: calculator              -> turns facts into numbers / decisions
  Writer      tools: none                    -> writes the customer reply and
                                                reviews it; can ask for more research

`create_agent` runs the classic loop for us: the LLM decides to call a tool ->
the tool runs -> the result goes back to the LLM -> repeat until it answers.
"""
from __future__ import annotations

import ast
import operator

from langchain.agents import create_agent
from langchain_core.tools import tool

import rag
from llm import build_llm


# ---------------------------------------------------------------- tools

@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF knowledge base. Returns the most relevant passages,
    each tagged [file p.N]. Use short, specific queries (e.g. "idle fee DC charger",
    "VoltPlus discount") and call it several times for different sub-topics."""
    return rag.format_hits(rag.search(query, k=5))


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("min", "max", "round"):
        return {"min": min, "max": max, "round": round}[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "42 * (24 + 2) * 0.85".
    Supports + - * / ** %, parentheses, min(), max(), round(). Always use this
    instead of doing maths in your head."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 4))
    except Exception as exc:
        return f"error: {exc}"


# ---------------------------------------------------------------- prompts

RESEARCHER_PROMPT = """You are the RESEARCHER on a customer-support team.
Your only job: find every fact in the company documents needed to handle the request.

- Break the request into sub-topics and call search_documents once per sub-topic
  (tariffs, surcharges, discounts, fees, refunds, SLAs, eligibility rules, taxes,
  minimums/penalties...). Make at least 4 searches; one search is never enough.
- Report facts VERBATIM-ish with exact numbers and conditions, each followed by its
  citation tag copied exactly from the search result, e.g.
  [VoltWay_Customer_Handbook_2026.pdf p.3] (real file name, never just "file").
- Do NOT calculate final amounts and do NOT write the customer reply.
- If something is not in the documents, say "NOT FOUND IN DOCUMENTS: <topic>".
Output a bullet list titled FACTS."""

ANALYST_PROMPT = """You are the ANALYST on a customer-support team.
You get the customer request and the Researcher's cited FACTS.

- For EVERY conditional rule (minimums, thresholds, tiers, time windows, eligibility)
  write: condition -> customer's value -> MET / NOT MET. Apply a charge, discount
  or credit ONLY if its condition is MET. Ignore rules for other customer types.
- Include taxes whenever the documents say they apply.
- Do every calculation with the calculator tool, step by step (no mental maths).
- Apply rules in the order the documents state (e.g. discount before tax).
- Keep citations next to each rule you use.
- Flag any missing fact you needed as "MISSING: <what>".
- If you receive REVIEWER FEEDBACK, fix exactly what it points out.
Output: RULE CHECKS, CALCULATION (step by step), RESULT."""

WRITER_PROMPT = """You are the WRITER and final REVIEWER on a customer-support team.
You get the customer request, the Researcher's FACTS and the Analyst's calculation.

First REVIEW the work critically against the FACTS:
- Was any charge/discount/credit applied whose condition is NOT actually met
  (e.g. a minimum-usage penalty when usage is above the minimum)? Was a required
  step skipped (e.g. tax)? Is a number inconsistent with the facts?
  -> reply with exactly one line: NEED_RECALCULATION: <what is wrong>
- Is a fact needed to answer the customer missing from FACTS?
  -> reply with exactly one line: NEED_MORE_RESEARCH: <precise thing to look up>
Reply with one of those lines and nothing else if there is a problem.
- Otherwise write the final reply to the customer: friendly, clear, concise, using
  the Analyst's numbers (do not recompute), a short itemised breakdown where useful,
  next steps (e.g. how to claim a refund), and a "Sources:" line listing the
  citations used. Never invent policy that is not in the facts."""


def build_agents():
    llm = build_llm()
    return {
        "researcher": create_agent(llm, tools=[search_documents], system_prompt=RESEARCHER_PROMPT),
        "analyst": create_agent(llm, tools=[calculator], system_prompt=ANALYST_PROMPT),
        "writer": create_agent(llm, tools=[], system_prompt=WRITER_PROMPT),
    }
