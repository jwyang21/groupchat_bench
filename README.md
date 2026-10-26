# GroupChat-Bench: Benchmarking LLM Agents on Long-Term Understanding of Realistic Group Chats

**A benchmark for evaluating LLM agents on long-term understanding of
realistic group chats.**

Code and data release for **GroupChat-Bench**, a benchmark that evaluates
LLM agents on long-term understanding of realistic, multi-party group-chat
histories. This repository also includes the reference implementation of
**GroupChat-Mem**, the memory system proposed alongside the benchmark, which
combines a structured **relational memory** with a **raw session repository**
and retrieves from both to answer questions over long group-chat histories.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)


* * *

## Overview

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
  records (open-schema - no predefined entity types or relation labels). Subjects and
  objects become nodes and relations become edges; each record carries its source
  session's timestamp. Records from different sessions are linked through shared nodes,
  forming a graph-structured relational memory. Re-extraction uses the *complete* partial
  session and **replaces** the previous extraction, so earlier records are revised as later
  utterances resolve references.
- **Raw session repository $\mathcal{S}$.** Each original session is stored verbatim, preserving exact
  wording, local discourse context, and conversational nuances that a compressed
  representation may omit.

Given a query, GroupChat-Mem retrieves evidence from **both** views in parallel and, if the
initial answer is insufficient, performs a fine-grained traversal over $\mathcal{R}$:

1. **From $\mathcal{R}$** - the LLM identifies the top-**n** query-relevant nodes; a **two-hop**
   neighborhood is built around each; the records are linearized, ranked by cosine
   similarity to the query (OpenAI embeddings), and the top-**k_r** records are kept.
2. **From $\mathcal{S}$** - each session is a retrieval unit; **k_s** sessions are retrieved,
   **half by BM25** (lexical) and **half by embedding cosine similarity** (semantic).
3. **Response** - the LLM generates an initial answer and judges its sufficiency. If
   sufficient, it is returned. Otherwise GroupChat-Mem traverses $\mathcal{R}$: it picks the most
   query-relevant node among the retrieved records, expands one hop at a time, and
   re-answers after each hop, stopping when the answer is supported or at the **maximum hop
   limit** - in which case it returns **"I don't know."**

$\mathcal{R}$ is induced purely from the observed dialogue; retrieval is reconstructed at each query's
timepoint so information from future sessions never leaks into a past question.

* * *

## Quick Starts

### 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

### 2. Environment Variables

- `OPENAI_API_KEY` - used for the OpenAI embeddings (`text-embedding-3-small`) in retrieval,
  and for OpenAI API backbones (e.g. GPT-5.x reasoning models).
- `GOOGLE_API_KEY` - used for Gemini backbones served through Gemini's OpenAI-compatible
  endpoint (`provider: openai`, `base_url: .../generativelanguage.../openai/`).
- A QA backbone. Any of:
  - a local **vLLM** endpoint (open models) via an OpenAI-compatible API - set the
    endpoint's host/port in the config (`llms[].host` / `llms[].port`) or via the
    `VLLM_HOST` / `VLLM_PORT` placeholders in the example configs; or
  - an **API model** (`provider: openai` with a `base_url`) - e.g. GPT-5.x
    (`configs/groupchat_mem/groupchat_mem.luna.friends.newname.t0.yaml`) or Gemini
    (`groupchat_mem.gemini.friends.newname.t0.yaml`). Reasoning models set `top_p: null`
    and control reasoning via `extra_body.reasoning_effort`.

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."        # only needed for Gemini backbones
```

**OpenAI-compatible gateway (optional).** The OpenAI path (embeddings + OpenAI API
backbones) can be routed through an OpenAI-compatible AI gateway instead of calling
OpenAI directly, via the `GROUPCHAT_API_MODE` environment variable. `openai` (default)
keeps the official OpenAI endpoint unchanged; `letsur` routes OpenAI-default requests to
the gateway (`https://gw.letsur.ai/v1`) using `LETSUR_API_KEY`. Model ids are identical
codes (no mapping), and vLLM / Gemini / any explicitly-set endpoint are never rerouted.

```bash
export GROUPCHAT_API_MODE="letsur"   # default "openai"; "letsur" uses the gateway below
export LETSUR_API_KEY="..."          # only needed when GROUPCHAT_API_MODE=letsur
```

**Cache model / QA model separation.** The memory cache (relational records, query
nodes, record embeddings) can be loaded from one model while QA is answered by another,
via `memory.cache_llm_name` (+ `memory.embedding_cache_readonly: true` to never write to
the source cache). This lets a frontier model answer over an open model's precomputed
cache without rebuilding it - see
`configs/groupchat_mem/groupchat_mem.terra_qwencache.friends.newname.t0.yaml`.

### 3. Run

**GroupChat-Mem (proposed method):**

```bash
PYTHONPATH=src python src/run/run_groupchat_mem.py \
  --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
  --method groupchat_mem --llms qwen3.5-35b-a3b-on
```

Larger open backbones have ready example configs too: `groupchat_mem.qwen122b.*`
(Qwen3.5-122B-A10B) and `groupchat_mem.gptoss120b.*` (GPT-OSS-120B). GPT-OSS is a
reasoning-effort model (no `thinking_token_budget`), so its config uses `top_p: 1.0`
and `extra_body.chat_template_kwargs.reasoning_effort: high`; swap `--llms` to match
the config's model name (`qwen3.5-122b-a10b` / `gpt-oss-120b`).

**A baseline (e.g. HippoRAG 2):**

```bash
PYTHONPATH=src python src/run/run_baseline.py \
  --config configs/baselines/hipporag2.qwen.friends.newname.t0.yaml \
  --llms qwen3.5-35b-a3b-on
```

* * *

## Components

### Repository structure

```
.
├── README.md
├── requirements.txt
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

Configs point at a `dataset_root` per show - `friends`, `bigbang`, `theoffice`. Each show
provides the chronological sessions and per-session candidate question sets. The `data_type`
selects a character-name setting - `newname` (New Names, the default), `normal` (Original
Names), or `shuffled` (the adversarial Swapped Names) - and `trial` selects one of the
three sampled question sets used for the reported mean/standard deviation.

The raw GroupChat-Bench data is available in this [link](https://drive.google.com/file/d/1WP-hNpbYM3uL8qkr8I8ouJnW0sfaCwu8/view?usp=sharing).

### GroupChat-Mem hyperparameters

| Hyperparameter | Value | Meaning |
|---|---|---|
| `num_records` (k_r) | 250 | records kept from the relational memory $\mathcal{R}$ |
| `num_sessions` (k_s) | 40 | sessions retrieved from the raw session repository $\mathcal{S}$ |
| `num_query_nodes` (n) | 3 | top-n query-relevant nodes identified in $\mathcal{R}$ |
| `hop` | 2 | two-hop neighborhood built around each query node |
| `max_hop` | 6 | maximum traversal hop limit |

### Baselines

| Setting | Description |
|---|---|
| **Direct QA (No History)** | Answers from parametric knowledge only (`simple_rag`, `no_history: true`). |
| **Simple RAG (BM25)** | Sparse lexical session retrieval (`simple_rag`, `retriever: bm25`). |
| **Simple RAG (OpenAI Embedding)** | Dense semantic session retrieval (`simple_rag`, `retriever: embedding`). |
| **HippoRAG 2** | OpenIE-based relational fact store + Personalized PageRank retrieval. |
| **Mem0** | Fact extraction with add/update/delete consolidation, dense retrieval. |
| **A-MEM** | Interlinked memory notes with dynamic linking and evolution. |
| **Oracle** | Upper bound with perfect retrieval and routing. |

Question format is selectable per config via `question_format`
(`multi_structured` / `multi_unstructured` / `open_ended`).

### Prompts

All prompt templates live in `prompts/`. Each is loaded by name (overridable per config
via the matching `prompt_*` key). QA prompts come in three question-format variants:
`*.txt` (multi_structured), `*.multi-unstructured.txt`, `*.open-ended.txt`.

| Prompt(s) | Used by | Purpose |
|---|---|---|
| `a_mem_note_construction.txt` | A-MEM | Build a memory note from a session (commit). |
| `a_mem_evolution.txt` | A-MEM | Evolve neighboring notes when a new note is linked. |
| `a_mem_link_judge.txt` | A-MEM | Decide whether two notes should be linked. |
| `a_mem_query_keywords.txt` | A-MEM | Extract query keywords at retrieval time. |
| `a_mem_qa.txt` | A-MEM | Final QA (A-MEM's own QA template). |
| `hipporag2_openie.txt` | HippoRAG 2 | 1-call OpenIE — extracts entities + triples in a single LLM call (the extraction path). |
| `hipporag2_filter.txt` | HippoRAG 2 | Filter retrieved facts for relevance. |
| `hipporag2_qa.txt` | HippoRAG 2 | Final QA (uses the sessions-view QA template). |
| `mem0_fact_extraction.txt` | Mem0 | Fact extraction (sentence-form facts — the reported setting). |
| `mem0_update_decision.txt` | Mem0 | Add / update / delete consolidation decision. |
| `qa_mem0.txt` | Mem0 | Final QA. |
| `qa_records.{,multi-unstructured,open-ended}.txt` | GroupChat-Mem, Oracle | QA over relational **records** (triples) only. |
| `qa_sessions.{,multi-unstructured,open-ended}.txt` | GroupChat-Mem, Oracle, Simple RAG | QA over retrieved **sessions** (dialogue) only. |
| `qa_dual.{,multi-unstructured,open-ended}.txt` | GroupChat-Mem, Oracle | Dual-view QA over records **and** sessions together. |
| `qa_no-history.{,multi-unstructured,open-ended}.txt` | Simple RAG (Direct QA) | QA from parametric knowledge only (no retrieved context). |
| `record_openie_extraction.txt` | GroupChat-Mem | OpenIE extraction that builds the relational memory $\mathcal{R}$ — turns each (partial) session into `[subject, relation, object]` records (run offline; records are then loaded at QA time). |
| `query_node_id.txt` | GroupChat-Mem | Identify the single query-relevant node in the record graph. |
| `query_node_id_select.txt` | GroupChat-Mem | Select the top-k query-relevant nodes. |
| `qa_traversal_lookup.txt` | GroupChat-Mem | During iterative traversal, decide the next entity to look up. |

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
| **GroupChat-Mem** | **77.8 / 77.6 / 64.8** | 77.5 / **78.9 / 70.3** | **78.7** / **77.1** / **69.0** |
