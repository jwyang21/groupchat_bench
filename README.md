<!-- Last update: 2026-08-03 -->
# GroupChat-Bench

**A dual-view memory for long-term understanding of realistic group chats.**

Reference implementation of **GroupChat-Mem**, the memory system proposed in
*"GroupChat-Bench: Benchmarking LLM Agents on Long-Term Understanding of Realistic
Group Chats."* GroupChat-Mem combines a structured **relational memory** with a
**raw session repository** and retrieves from both to answer questions over long,
multi-party group-chat histories.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)


* * *

## Overview 📚

Group chats pose challenges that most dialogue benchmarks do not capture: **multi-party
grounding** (who an utterance concerns and how it fits the relationships among
participants), **temporal persistence** (personal and relational information accumulates,
changes, and is reinterpreted across sessions), and **unstructured interaction**
(fragmented utterances, pile-on reactions, and interleaved topics). Agents are evaluated
as sessions unfold chronologically, answering from only the conversation history available
when each question is posed.

**GroupChat-Mem** is a dual-view memory built online from the dialogue only:

- **Relational memory $\mathcal{R}$.** At each turn, an LLM performs **Open Information Extraction
  (OpenIE)** over the (partial) session and produces **`[subject, relation, object]`**
  records (open-schema — no predefined entity types or relation labels). Subjects and
  objects become nodes and relations become edges; each record carries its source
  session's timestamp. Records from different sessions are linked through shared nodes,
  forming a graph-structured relational memory. Re-extraction uses the *complete* partial
  session and **replaces** the previous extraction, so earlier records are revised as later
  utterances resolve references.
- **Raw session repository $\mathcal{S}$.** Each original session is stored verbatim, preserving exact
  wording, local discourse context, and conversational nuances that a compressed
  representation may omit.

Given a query, GroupChat-Mem retrieves evidence from **both** views in parallel and, if the
initial answer is insufficient, performs a fine-grained traversal over R:

1. **From $\mathcal{R}$** — the LLM identifies the top-**n** query-relevant nodes; a **two-hop**
   neighborhood is built around each; the records are linearized, ranked by cosine
   similarity to the query (OpenAI embeddings), and the top-**k_r** records are kept.
2. **From $\mathcal{S}$** — each session is a retrieval unit; **k_s** sessions are retrieved,
   **half by BM25** (lexical) and **half by embedding cosine similarity** (semantic).
3. **Response** — the LLM generates an initial answer and judges its sufficiency. If
   sufficient, it is returned. Otherwise GroupChat-Mem traverses R: it picks the most
   query-relevant node among the retrieved records, expands one hop at a time, and
   re-answers after each hop, stopping when the answer is supported or at the **maximum hop
   limit** — in which case it returns **"I don't know."**

R is induced purely from the observed dialogue; retrieval is reconstructed at each query's
timepoint so information from future sessions never leaks into a past question.

* * *

## Quick Starts 🚀

### 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

### 2. Environment Variables

- `OPENAI_API_KEY` — used for the OpenAI embeddings (`text-embedding-3-small`) in retrieval.
- A local **vLLM** endpoint serving the QA backbone via an OpenAI-compatible API. Set the
  endpoint's host/port in the config (`llms[].host` / `llms[].port`) or via the
  `VLLM_HOST` / `VLLM_PORT` placeholders shown in the example configs.

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run

**GroupChat-Mem (proposed method):**

```bash
PYTHONPATH=src python src/run/run_groupchat_mem.py \
  --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
  --method groupchat_mem --llms qwen3.5-35b-a3b-on
```

**A baseline (e.g. HippoRAG 2):**

```bash
PYTHONPATH=src python src/run/run_baseline.py \
  --config configs/baselines/hipporag2.qwen.friends.newname.t0.yaml \
  --llms qwen3.5-35b-a3b-on
```

* * *

## Components ⚙️

### Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   ├── groupchat_mem/    # the proposed method
│   └── baselines/        # a_mem, mem0, hipporag2, oracle, simple_rag / direct_qa
├── prompts/              # prompt templates (OpenIE, QA, traversal, baselines)
└── src/
    ├── groupchat_qa/
    │   ├── client/       # LLM clients (OpenAI-compatible / vLLM), load-aware routing
    │   ├── embedder/     # embedders + on-disk embedding cache
    │   ├── memory/       # groupchat_mem + a_mem, mem0, hipporag2, oracle, simple_rag
    │   ├── pipeline/     # data loading
    │   └── utils/        # tokenization, formatting, filesystem helpers
    └── run/
        ├── run_groupchat_mem.py   # entry: proposed method (grid / resume)
        └── run_baseline.py        # entry: baselines + oracle (commit + QA)
```

### Data layout

Configs point at a `dataset_root` per show — `friends`, `bigbang`, `theoffice`. Each show
provides the chronological sessions and per-session candidate question sets. The `data_type`
selects a character-name setting — `newname` (New Names, the default), `normal` (Original
Names), or `shuffled` (the adversarial Swapped Names) — and `trial` selects one of the
three sampled question sets used for the reported mean/standard deviation.

The raw GroupChat-Bench data is available in this [link](https://drive.google.com/file/d/1-MjPm9bH7-0lNLl9cG1U_KKdU-t2BISO/view?usp=drive_link).

### GroupChat-Mem hyperparameters

| Hyperparameter | Value | Meaning |
|---|---|---|
| `num_records` (k_r) | 40 | records kept from the relational memory R |
| `num_sessions` (k_s) | 250 | sessions retrieved from the raw session repository S |
| `num_query_nodes` (n) | 3 | top-n query-relevant nodes identified in R |
| `hop` | 2 | two-hop neighborhood built around each query node |
| `max_hop` | 6 | maximum traversal hop limit |

### Baselines

| Setting | Description |
|---|---|
| **Direct QA (No History)** | Answers from parametric knowledge only (`simple_rag`, `no_history: true`). |
| **Simple RAG (BM25)** | Sparse lexical session retrieval (`simple_rag`, `retriever: bm25`). |
| **Simple RAG (OpenAI Embedding)** | Dense semantic session retrieval (`simple_rag`, `retriever: embedding`). |
| **HippoRAG 2** | OpenIE knowledge graph + Personalized PageRank retrieval. |
| **Mem0** | Fact extraction with add/update/delete consolidation, dense retrieval. |
| **A-MEM** | Interlinked memory notes with dynamic linking and evolution. |
| **Oracle** | Upper bound with perfect retrieval and routing. |

Question format is selectable per config via `question_format`
(`multi_structured` / `multi_unstructured` / `open_ended`).

* * *

## Results

Main results on the New Names setting (mean over three runs), reported by the paper across
three backbones (Qwen3.5-35B-A3B, Gemma4-26B-A4B, GPT-OSS 20B); the best memory system per
column is in **bold**. Each cell shows accuracy (%) as Qwen3.5 / Gemma4 / GPT-OSS.

| Memory System | Friends | The Big Bang Theory | The Office |
|---|---|---|---|
| Direct QA (No History) | 21.4 / 28.7 / 24.6 | 22.5 / 27.3 / 29.6 | 28.2 / 32.3 / 30.3 |
| Simple RAG (BM25) | 69.9 / 69.3 / 56.2 | 69.7 / 68.9 / 61.5 | 65.8 / 64.5 / 59.6 |
| Simple RAG (OpenAI Embedding) | 70.4 / 70.2 / 58.7 | 74.9 / 76.0 / 67.8 | 74.3 / 74.6 / 67.0 |
| HippoRAG 2 | 73.2 / 73.0 / 60.1 | 78.1 / 77.4 / 68.1 | 75.1 / 75.3 / 68.5 |
| Mem0 | 71.2 / 71.5 / 62.3 | 73.9 / 71.4 / 64.9 | 68.8 / 69.9 / 66.0 |
| A-MEM | 65.2 / 69.2 / 58.0 | 69.6 / 70.2 / 61.8 | 68.1 / 69.3 / 62.7 |
| **GroupChat-Mem** | **77.8 / 77.6 / 64.8** | 77.5 / **78.9 / 70.3** | **78.7** / 77.1 / **69.0** |

* * *

## License

Released under the [MIT License](LICENSE).

* * *

## Citation

```bibtex
@inproceedings{groupchatbench,
  title     = {GroupChat-Bench: Benchmarking LLM Agents on Long-Term Understanding of Realistic Group Chats},
  author    = {Anonymous},
  booktitle = {Anonymous submission},
  year      = {2026}
}
```
