# LangChain Multi-Agent RAG: VoltWay Support Desk

Three LangChain agents work as a team, coordinated by LangGraph, to answer customer
questions using PDFs you upload (RAG).

**Case example:** VoltWay, a fictional EV-charging company. Customers ask billing and
refund questions; the answers depend on rules spread across two PDFs (customer
handbook + fleet agreement).

```
                 +------- NEED_MORE_RESEARCH -----------------+
                 |              +--- NEED_RECALCULATION ------+
                 v              v                             |
  START -> [Researcher] -> [Analyst] -> [Writer/Reviewer] ----+--> END
              |                |
       search_documents    calculator
         (RAG over PDFs)   (exact maths)
```

| Agent | Tool | Job |
|---|---|---|
| 🔎 Researcher | `search_documents` (vector search over the PDFs) | Finds every relevant rule, with `[file.pdf p.N]` citations |
| 🧮 Analyst | `calculator` | Checks which rules apply (MET / NOT MET), does the maths step by step |
| ✍️ Writer | none | Reviews the work; sends it back if wrong or incomplete, otherwise writes the reply |

- **LLM:** LLMAPI gateway used by all agents.
- **Embeddings:** local all-MiniLM-L6-v2 (bundled with chromadb, runs on CPU, no API).
- **Vector DB:** Chroma, saved on disk in `storage/chroma`.

## Setup (once)

```bash
cd ~/Desktop/langchain_multiagent_rag
python3 -m venv .venv            # already done
.venv/bin/pip install -r requirements.txt
```
The LLMAPI gateway must be running. Override with env vars

## Use it

```bash
source .venv/bin/activate

python main.py demo                  # upload the sample PDFs + run both example cases
python main.py demo --case 1         # just the VoltPlus customer case
python main.py upload my.pdf b.pdf   # upload your own PDFs
python main.py docs                  # list what's in the knowledge base
python main.py ask "your question"   # one question
python main.py chat                  # keep asking questions
python main.py reset                 # wipe the vector DB

streamlit run app.py                 # web UI: upload PDFs + ask, in the browser
```

### Uploading a PDF
- **CLI:** `python main.py upload /path/to/file.pdf`. The file is copied to `data/pdfs/`,
  split into chunks, embedded and stored. Uploading the same file name again replaces it.
- **Web UI:** `streamlit run app.py`, then in the sidebar use "Choose PDF files" and
  click **Upload & index**.
- PDFs must contain text. Scanned PDFs (images only) need OCR first.

## Example cases (both verified)

1. **VoltPlus customer:** 42 kWh, Saturday, highway ultra-fast charger, 35 min idle,
   plus a failed attempt charged ₹150.
   → 42×(24+2)=1092 → −15% = 928.20 → +idle (35−20)×5 = 75 → +18% GST = **₹1,183.78**;
   the ₹150 is **refunded automatically** (failed within 5 min due to a charger fault).
2. **Fleet partner:** 1,600 kWh at the depot at night, a charger down for 20 h.
   → 1600×11 = 17,600 → −10% tier = 15,840 → +GST = **₹18,691.20**; SLA broken (12 h),
   so a **₹792 credit** (5% of pre-GST) goes on the next invoice. No shortfall charge,
   because 1,600 is above the 1,000 kWh minimum (the Reviewer catches this if the Analyst gets it wrong).

## Files

| File | What it is |
|---|---|
| `llm.py` | LLMAPI chat model + local embedding model |
| `rag.py` | PDF → chunks → Chroma; `search()` for retrieval |
| `agents.py` | the 3 agents: prompts + tools |
| `graph.py` | LangGraph workflow: state, nodes, review loop |
| `main.py` | CLI |
| `app.py` | Streamlit web UI |
| `make_sample_pdfs.py` | generates the VoltWay sample PDFs |
