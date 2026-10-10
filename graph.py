"""
The multi-agent workflow, built with LangGraph.

                 +------- NEED_MORE_RESEARCH -----------------+
                 |              +--- NEED_RECALCULATION ------+
                 v              v                             |
  START -> [researcher] -> [analyst] -> [writer] --> done? ---+--> END

The Writer doubles as a reviewer: it can bounce work back to either agent
(at most MAX_REVISIONS times in total), the way a senior reviews a junior's draft.

LangGraph concepts:
  * STATE  - a typed dict passed between nodes (like a shared "case file").
  * NODE   - a function that reads the state and returns the fields it changed.
  * EDGE   - which node runs next. A *conditional* edge picks at runtime.

Each node here wraps one agent: it builds the message for that agent from the
state, runs it, and writes the agent's output back into the state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agents import build_agents

MAX_REVISIONS = 2


class CaseState(TypedDict, total=False):
    question: str
    facts: str            # Researcher output (accumulates across rounds)
    analysis: str         # Analyst output
    answer: str           # Writer output (final reply)
    research_request: str  # Writer -> Researcher: what to look up
    review_feedback: str   # Writer -> Analyst: what to fix
    rounds: int            # researcher runs
    revisions: int         # times the Writer sent work back
    # every node appends to this log; operator.add tells LangGraph to concatenate
    trace: Annotated[list[dict], operator.add]


def _run(agent, prompt: str) -> tuple[str, list[str]]:
    """Run one agent, return (final text, list of tool calls it made)."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    calls = []
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            calls += [f"{c['name']}({next(iter(c['args'].values()), '')!r})" for c in m.tool_calls]
        elif isinstance(m, ToolMessage) and m.content.startswith("error"):
            calls.append(f"  -> {m.content}")
    text = result["messages"][-1].content
    if isinstance(text, list):  # some providers return content blocks
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    return text.strip(), calls


def build_graph(on_event: Callable[[dict], None] | None = None):
    agents = build_agents()
    emit = on_event or (lambda e: None)

    def researcher(state: CaseState) -> dict:
        rounds = state.get("rounds", 0) + 1
        if state.get("research_request"):
            prompt = (f"Customer request:\n{state['question']}\n\n"
                      f"Facts already found:\n{state.get('facts', '')}\n\n"
                      f"The reviewer needs MORE information on: {state['research_request']}\n"
                      "Search for it and list only the NEW facts.")
        else:
            prompt = f"Customer request:\n{state['question']}"
        emit({"agent": "researcher", "status": "start", "round": rounds})
        text, calls = _run(agents["researcher"], prompt)
        facts = (state.get("facts", "") + "\n\n" + text).strip() if state.get("facts") else text
        event = {"agent": "researcher", "round": rounds, "tools": calls, "output": text}
        emit(event)
        return {"facts": facts, "rounds": rounds, "research_request": "", "trace": [event]}

    def analyst(state: CaseState) -> dict:
        prompt = f"Customer request:\n{state['question']}\n\nFACTS:\n{state['facts']}"
        if state.get("review_feedback"):
            prompt += (f"\n\nYOUR PREVIOUS ANALYSIS:\n{state['analysis']}"
                       f"\n\nREVIEWER FEEDBACK:\n{state['review_feedback']}")
        emit({"agent": "analyst", "status": "start", "revision": bool(state.get("review_feedback"))})
        text, calls = _run(agents["analyst"], prompt)
        event = {"agent": "analyst", "tools": calls, "output": text}
        emit(event)
        return {"analysis": text, "review_feedback": "", "trace": [event]}

    def writer(state: CaseState) -> dict:
        emit({"agent": "writer", "status": "start"})
        text, calls = _run(agents["writer"],
                           f"Customer request:\n{state['question']}\n\nFACTS:\n{state['facts']}\n\n"
                           f"ANALYST:\n{state['analysis']}")
        event = {"agent": "writer", "tools": calls, "output": text}
        emit(event)
        revisions = state.get("revisions", 0)
        if revisions < MAX_REVISIONS:
            if text.startswith("NEED_MORE_RESEARCH:"):
                return {"research_request": text.split(":", 1)[1].strip(),
                        "revisions": revisions + 1, "trace": [event]}
            if text.startswith("NEED_RECALCULATION:"):
                return {"review_feedback": text.split(":", 1)[1].strip(),
                        "revisions": revisions + 1, "trace": [event]}
        elif text.startswith(("NEED_MORE_RESEARCH:", "NEED_RECALCULATION:")):
            # revision limit reached: stop and return the best analysis we have
            return {"answer": f"⚠️ Reviewer still flagged: {text}\n\nBest available analysis:\n"
                              f"{state.get('analysis', '')}", "trace": [event]}
        return {"answer": text, "trace": [event]}

    def route_after_writer(state: CaseState) -> str:
        if state.get("answer"):
            return END
        if state.get("research_request"):
            return "researcher"
        if state.get("review_feedback"):
            return "analyst"
        return END

    g = StateGraph(CaseState)
    g.add_node("researcher", researcher)
    g.add_node("analyst", analyst)
    g.add_node("writer", writer)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "writer")
    g.add_conditional_edges("writer", route_after_writer, ["researcher", "analyst", END])
    return g.compile()


def ask(question: str, on_event: Callable[[dict], None] | None = None) -> CaseState:
    return build_graph(on_event).invoke({"question": question, "rounds": 0, "revisions": 0, "trace": []})
