# GroupChat-Bench: Benchmarking LLM Agents on Long-Term Understanding of Realistic Group Chats

**A benchmark for evaluating LLM agents on long-term understanding of
realistic group chats.**

Code and data release for **GroupChat-Bench**, a benchmark that evaluates
LLM agents on long-term understanding of realistic, multi-party group-chat
histories. This repository also includes the reference implementation of
**GroupChat-Mem**, the memory system proposed alongside the benchmark, which
combines a structured **relational memory** with a **raw session repository**
and retrieves from both to answer questions over long group-chat histories.

> **Note:** An earlier version of this work, titled **DialSim**, was presented
> at the **ICLR 2026 MemAgents Workshop**. **GroupChat-Bench** is an extended
> and updated version of that work.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)

* * *

## Overview

Group chats pose challenges that most dialogue benchmarks do not capture: **multi-party
grounding** (who an utterance concerns and how it fits the relationships among
participants), **temporal persistence** (personal and relational information accumulates,
changes, and is reinterpreted across sessions), and **entangled interaction**
(fragmented utterances, pile-on reactions, and interleaved topics). Agents are evaluated
as sessions unfold chronologically, answering from only the conversation history available
when each question is posed.

**GroupChat-Mem** is a dual-view memory built from the dialogue only:

- **Relational memory $\mathcal{R}$.** An LLM performs **Open Information Extraction
  (OpenIE)** over the (partial) session and produces **`[subject, relation, object]`**
  records (open-schema - no predefined entity types or relation labels). In this release
  the extraction is run twice per session: over the partial session up to the question
  turn (used for that session's question) and over the complete session (used for every
  later question). Subjects and
  objects become nodes and relations become edges; each record carries its source
  session's timestamp. Records from different sessions are linked through shared nodes,
  forming a graph-structured relational memory. Each extraction reads the whole (partial)
  session, and the complete-session extraction **replaces** the partial one, so earlier
  records are revised as later utterances resolve references.
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

$\mathcal{R}$ is induced purely from the observed dialogue. The extractions are precomputed for
speed, but the causal condition is the same as running online: a question in session *i* sees
only the complete-session records of sessions before *i* and the partial-session records of
session *i* up to the question turn, so information from later sessions never leaks into a
past question.

* * *

## Quick Start

The whole pipeline in four steps. Each step is explained in detail in the sections below.

```bash
# 1. install
git clone https://github.com/jwyang21/groupchat_bench.git
cd groupchat_bench
python -m venv .venv && source .venv/bin/activate        # Python >= 3.10
pip install -r requirements.txt

# 2. get the ready-to-run data (creates ./data/; see "Download the Data" below)
pip install gdown
gdown 12qMDnfCd8eZjiGKSU9L0UuoF67Yg8zI1 -O groupchat_bench_processed.zip
unzip groupchat_bench_processed.zip

# 3. keys + a QA backbone (here: a local vLLM server, run it in a separate terminal)
export OPENAI_API_KEY="sk-..."                           # embeddings
vllm serve Qwen/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b-a3b-on --port 8000 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
sed -i 's/<VLLM_HOST>/localhost/; s/<VLLM_PORT>/8000/' \
  configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml

# 4. run GroupChat-Mem on Friends / New Names / trial 0
PYTHONPATH=src python src/run/run_groupchat_mem.py \
  --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
  --method groupchat_mem --llms qwen3.5-35b-a3b-on
```

* * *

## 1. Installation

```bash
git clone https://github.com/jwyang21/groupchat_bench.git
cd groupchat_bench
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

`requirements.txt` covers both the experiments and the preprocessing scripts. A vLLM
installation is only needed on the machine that serves open models.

## 2. The GroupChat-Bench Data

### 2.1 What is in the benchmark

GroupChat-Bench is built from the scripts of three sitcoms. Each scene is one group-chat
**session**; sessions are ordered chronologically and each carries its air date.

| Show (`<show>`) | Sessions | Assistant character |
|---|---|---|
| Friends (`friends`) | 788 | Ross |
| The Big Bang Theory (`bigbang`) | 805 | Sheldon |
| The Office (`theoffice`) | 2347 | Michael |

- **One question per session.** In session *i* a participant (the *questioner*) asks the
  assistant character a five-choice question in the middle of the conversation. The agent
  may use sessions 0 .. *i*-1 plus session *i* up to the question turn; nothing later.
- **Question types** (ratio 1 : 1 : 2): *TKG 1-hop* and *TKG 2-hop* questions about
  relations between characters, and *fan-quiz* questions about events in the show. A TKG
  question is asked either after the relation's date (past) or on that date (current). If
  no prior evidence exists at the question time (e.g., the relation only appears in a
  later session), the gold answer is (E) "I don't know." The raw time-type labels
  (`past`, `cur`, `fu`, ...) mark these cases.
- **Five options.** Option (E) is always "I don't know. (None of the above)". The correct
  letter is balanced over (A)-(E), so chance accuracy is 20%.
- **Name settings** (`data_type`): `normal` (original character names), `newname`
  (characters consistently renamed, so parametric knowledge of the show does not help - the
  main setting of the paper) and `shuffled` (character names swapped with each other, an
  adversarial setting).
- **Trials** (`trial` 0 / 1 / 2): three independently sampled question sets; the paper
  reports the mean over the three.

Config values and the names used in the paper:

| Config key | Config value | Paper name |
|---|---|---|
| `data_type` | `normal` | Original Names |
| `data_type` | `newname` | New Names |
| `data_type` | `shuffled` | Swapped Names |
| `question_format` | `multi_structured` | Labeled MCQ |
| `question_format` | `multi_unstructured` | Naturalized MCQ |
| `question_format` | `open_ended` | Open-Ended |

### 2.2 Download the Data

Two archives are provided. **Most users only need the processed data.**

| Archive | Size | Contents | Use it to |
|---|---|---|---|
| **Processed data** `groupchat_bench_processed.zip` - [download](https://drive.google.com/file/d/12qMDnfCd8eZjiGKSU9L0UuoF67Yg8zI1/view?usp=sharing) | ~155 MB (1.2 GB unzipped) | Everything the experiments read, for all shows / name settings / trials, including the records extracted by the three paper backbones | run every experiment directly |
| Raw data `groupchat_bench.zip` - [download](https://drive.google.com/file/d/1WP-hNpbYM3uL8qkr8I8ouJnW0sfaCwu8/view?usp=sharing) | ~390 MB (4.5 GB unzipped) | The original question pools and oracle annotations (pickles) | regenerate the processed data yourself ([Section 3](#3-building-the-inputs-from-the-raw-data)) |

```bash
pip install gdown   # optional: command-line download from Google Drive
# processed data -> creates ./data/
gdown 12qMDnfCd8eZjiGKSU9L0UuoF67Yg8zI1 -O groupchat_bench_processed.zip
unzip groupchat_bench_processed.zip
# raw data (only for Section 3) -> creates ./groupchat_bench/
gdown 1WP-hNpbYM3uL8qkr8I8ouJnW0sfaCwu8 -O groupchat_bench.zip
unzip groupchat_bench.zip
```

All example configs read `./data/<show>` (`data.dataset_root: ./data/friends`). To keep the
data elsewhere, change `dataset_root` in the config. Sanity check:
`ls data/friends/date.json data/friends/newname/precomputed/per-trial/t0/qa.json`.

### 2.3 Processed data layout

```
data/
├── gt_triples/<show>.json                         # LLM-extracted, author-verified records of every session
└── <show>/                                         # friends / bigbang / theoffice
    ├── date.json                                   # air date of each session
    └── <data_type>/                                # normal / newname / shuffled
        ├── entire_session.json                     # full dialogue of each session
        ├── precomputed/
        │   ├── common/openie/<llm>/b06000/
        │   │   └── entire_session_records_raw.json # records of each full session
        │   └── per-trial/t<N>/                     # N = 0, 1, 2
        │       ├── qa.json                         # the question of each session
        │       ├── partial_dialogues.json          # each session cut at its question turn
        │       ├── qa_provenance.json              # (normal only) source of each question
        │       └── per-llm/<llm>/b06000/
        │           ├── openie/partial_session_records_raw.json   # records of each partial session
        │           ├── query_node_raw_nq3.json                   # query nodes of each question
        │           └── query_node_raw_nq3_attempted.json
        └── oracle/                                 # newname only
            ├── trial<N>.oracle.dialogue.json       # answer sessions of each question
            └── <llm>/trial<N>.oracle_raw.json      # records of those answer sessions
```

`<llm>` is `qwen3.5-35b-a3b-on`, `gemma-4-26b-on` or `gpt-oss-20b` (the paper backbones,
records extracted by the model itself) or `gt` (LLM-extracted, author-verified records, used by the
Oracle). `b06000` stands for the 6000-token reasoning budget used while extracting.
All lists are indexed by session (index *i* = session *i*).

| File | Format |
|---|---|
| `date.json` | `["September 22, 1994", ...]` |
| `entire_session.json`, `partial_dialogues.json` | `["Monica: There's nothing to tell! ...", ...]` |
| `qa.json` | column lists: `question`, `option` (5 strings), `true_answer`, `true_answer_choice` (`"(A)"`..`"(E)"`), `questioner`, `tkg_hop` (`1`, `2`, or `-1` for fan-quiz) |
| `*_records_raw.json` | `[[{"head", "relation", "tail", "start_date"}, ...], ...]` - one list of records per session; `start_date` is the session's date |
| `query_node_raw_nq3.json` | `{"<session index>": ["node", ...]}` - the top-3 record-graph nodes the question is about |
| `oracle/*.json` | one entry per question with its `timestamp` (dates of the answer sessions) and the evidence: `oracle_dialogues_time_ordered` / `_by_bm25` / `_by_cosine` / `_by_alternating`, or `oracle_records_by_cosine` |
| `gt_triples/<show>.json` | `[{"date": ..., "triples": [[head, relation, tail], ...]}, ...]` (lower-case, original names) |

What each method reads:

- **Baselines** (Direct QA, Simple RAG, HippoRAG 2, Mem0, A-MEM) read only `date.json`,
  the dialogues and `qa.json` (`per_llm_precomputed: false`) and build their own memories
  at run time.
- **Oracle** additionally reads `oracle/` (dialogues, and `gt` records for the record
  variants).
- **GroupChat-Mem** (`per_llm_precomputed: true`) also reads the records and query nodes of
  the backbone named in `llms[].name`. They ship for the three paper backbones. For any
  other backbone (e.g. the `qwen122b`, `gptoss120b`, `luna`, `gemini` example configs),
  build them first with steps 2-3 of [Section 3](#3-building-the-inputs-from-the-raw-data),
  or reuse a shipped backbone's records with `memory.cache_llm_name` (see the `terra_qwencache`
  config). A missing file raises `FileNotFoundError` with the expected path.
- A few `query_node_raw_nq3.json` files of the `normal` / `shuffled` settings do not cover
  every question; the missing questions are identified live during the run with the same
  prompts (`prompts/query_node_id*.txt`). Embeddings are not shipped: they are computed with the OpenAI API on the
  first run and cached under `data/` (`*_embedding_cache.pickle`).

### 2.4 Raw data layout

```
groupchat_bench/
├── <show>_groupchat_bench.pickle   # sessions: transcript ("script") + question pools
├── <show>_oracle_tkg.pickle        # answer-session dates of every TKG question
└── <show>_oracle_fan.pickle        # answer-session dates of every fan-quiz question
```

`<show>_groupchat_bench.pickle` is `{episode: {scene: session}}`; each session holds
`script` (the transcript), `hard_q` (TKG questions by time type, e.g. `past`, `cur_fu`)
and `easy_q` (fan-quiz questions). Every question has `questions` (the wording for each
possible questioner), `options` and `answer`. The LLM-extracted, author-verified records are in the
processed archive (`data/gt_triples/`).

* * *

## 3. Building the Inputs from the Raw Data

The `preprocess/` scripts turn the raw pickles into exactly the processed layout above.
You only need them to rebuild the data, to change the sampling, or to prepare records for a
new backbone. All scripts are resumable (re-running the same command continues).

| Step | Script | Needs | Produces |
|---|---|---|---|
| 1 | `step1_build_text.py` | raw pickles | `date.json`, dialogues, `qa.json`, `partial_dialogues.json`, `qa_provenance.json` |
| 4 | `step4_convert_gt_records.py` | `gt_triples/` | `gt` records for the Oracle |
| 2 | `step2_extract_records.py` | step 1, an LLM endpoint | the backbone's records (entire + partial sessions) |
| 3 | `step3_build_query_nodes.py` | step 2, an LLM endpoint | the backbone's query nodes |
| 5 | `step5_build_oracle.py` | steps 1, 4 (and 2) | `oracle/` bundles |

Run them from the repository root, in the order shown (step 4 is independent of steps 2-3):

```bash
RAW=./groupchat_bench      # unzipped raw archive
DATA=./data                # output root (the configs' dataset_root parent)

# Step 1 - sample one question per session (seeded by the trial, deterministic), cut each
#          session at the question turn, write all three name settings.
python preprocess/step1_build_text.py --raw-dir $RAW --data-root $DATA

# Step 4 - author-verified gt records -> gt record files (format + renaming only).
python preprocess/step4_convert_gt_records.py --gt-dir $DATA/gt_triples --data-root $DATA

# Step 2 - LLM record extraction (prompts/record_openie_extraction.txt) from each setting's
#          own dialogues; every record gets its session's date. Needs an OpenAI-compatible
#          endpoint serving the model under --llm (see the vLLM command in Quick Start).
python preprocess/step2_extract_records.py --data-root $DATA \
  --llm qwen3.5-35b-a3b-on --base-url http://localhost:8000/v1 --workers 32

# Step 3 - query-node identification with the same code GroupChat-Mem runs at query time
#          (prompts/query_node_id*.txt); the endpoint/sampling come from a GroupChat-Mem config.
PYTHONPATH=src python preprocess/step3_build_query_nodes.py --data-root $DATA \
  --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
  --llm qwen3.5-35b-a3b-on --base-url http://localhost:8000/v1

# Step 5 - oracle bundles (answer sessions + their records) for the Oracle baselines.
python preprocess/step5_build_oracle.py --raw-dir $RAW --data-root $DATA \
  --data-types newname --record-llms gt
```

Notes:

- Every script accepts `--shows`, `--data-types` and `--trials` (comma lists) to restrict
  the work, e.g. `--shows friends --data-types newname --trials 0`.
- Step 2 sampling presets (`--preset auto` picks one from the model name): thinking models
  (Qwen3.5 / Gemma-4) use temperature 0.6, top_p 0.95 and a 6000-token reasoning budget;
  GPT-OSS uses `reasoning_effort: medium`. The vLLM server of a thinking model needs
  `--reasoning-parser` and `--reasoning-config` for the budget to apply.
- Step 3 reuses the `llms` entry of the given config with the extraction-time reasoning
  setting (budget 6000 / `reasoning_effort: medium`). `--base-url` overrides the config's
  endpoint. A question that fails or times out is not written and is retried on the next run.
- Step 5 computes the cosine orderings with OpenAI embeddings (`OPENAI_API_KEY`); pass
  `--no-embedding` to skip them (the paper's Oracle uses the time-ordered dialogues, which
  do not need embeddings). Add backbone names to `--record-llms` to also bundle their
  records, e.g. `--record-llms gt,qwen3.5-35b-a3b-on`.
- LLM sampling is stochastic, so steps 2-3 reproduce the procedure, not byte-identical
  files; the shipped processed data are the files used in the paper.

* * *

## 4. Environment Variables

- `OPENAI_API_KEY` - used for the OpenAI embeddings (`text-embedding-3-small`) in retrieval,
  and for OpenAI API backbones (e.g. GPT-5.x reasoning models).
- `GOOGLE_API_KEY` - used for Gemini backbones served through Gemini's OpenAI-compatible
  endpoint (`provider: openai`, `base_url: .../generativelanguage.../openai/`).
- A QA backbone. Any of:
  - a local **vLLM** endpoint (open models) via an OpenAI-compatible API - set the
    endpoint's host/port in the config (`llms[].host` / `llms[].port`) or via the
    `<VLLM_HOST>` / `<VLLM_PORT>` placeholders in the example configs; or
  - an **API model** (`provider: openai` with a `base_url`) - e.g. GPT-5.x
    (`configs/groupchat_mem/groupchat_mem.luna.friends.newname.t0.yaml`) or Gemini
    (`groupchat_mem.gemini.friends.newname.t0.yaml`). Reasoning models set `top_p: null`
    and control reasoning via `extra_body.reasoning_effort`.

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."        # only needed for Gemini backbones
```

**OpenAI-compatible gateway (optional).** The OpenAI path (embeddings + OpenAI API
backbones) can be routed through any OpenAI-compatible gateway instead of calling
OpenAI directly, via the `GROUPCHAT_API_MODE` environment variable. `openai` (default)
keeps the official OpenAI endpoint unchanged; `gateway` routes OpenAI-default requests to
`GATEWAY_BASE_URL` using `GATEWAY_API_KEY`. Model ids are passed through unchanged
(the gateway must expose the same ids), and vLLM / Gemini / any explicitly-set endpoint
are never rerouted.

```bash
export GROUPCHAT_API_MODE="gateway"              # default "openai"
export GATEWAY_BASE_URL="https://<gateway>/v1"   # only needed in gateway mode
export GATEWAY_API_KEY="..."                     # only needed in gateway mode
```

**Cache model / QA model separation.** The memory cache (relational records, query
nodes, record embeddings) can be loaded from one model while QA is answered by another,
via `memory.cache_llm_name` (+ `memory.embedding_cache_readonly: true` to never write to
the source cache). This lets a frontier model answer over an open model's precomputed
cache without rebuilding it - see
`configs/groupchat_mem/groupchat_mem.terra_qwencache.friends.newname.t0.yaml`.

## 5. Run the Experiments

Every example config uses Friends / New Names / trial 0. To run another setting, edit
`data.dataset_root` (`./data/{friends,bigbang,theoffice}`), `data.data_type`
(`newname` / `normal` / `shuffled`) and `data.trial` (`0` / `1` / `2`), and give each run
its own output path.

For a local vLLM backbone, serve the model under the name used in the config
(`--served-model-name qwen3.5-35b-a3b-on` for the Qwen examples) and replace the
`<VLLM_HOST>` / `<VLLM_PORT>` placeholders, e.g.:

```bash
sed -i 's/<VLLM_HOST>/localhost/; s/<VLLM_PORT>/8000/' \
  configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml
```

**GroupChat-Mem (proposed method):**

```bash
PYTHONPATH=src python src/run/run_groupchat_mem.py \
  --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
  --method groupchat_mem --llms qwen3.5-35b-a3b-on
```

Larger open backbones have ready example configs too: `groupchat_mem.qwen122b.*`
(Qwen3.5-122B-A10B) and `groupchat_mem.gptoss120b.*` (GPT-OSS-120B). Their records are not
shipped, so build them first (steps 2-3 of Section 3). GPT-OSS is a reasoning-effort model
(no `thinking_token_budget`), so its config uses `top_p: 1.0` and
`extra_body.chat_template_kwargs.reasoning_effort: high`; swap `--llms` to match the
config's model name (`qwen3.5-122b-a10b` / `gpt-oss-120b`).

**A baseline (e.g. HippoRAG 2):**

```bash
PYTHONPATH=src python src/run/run_baseline.py \
  --config configs/baselines/hipporag2.qwen.friends.newname.t0.yaml \
  --llms qwen3.5-35b-a3b-on \
  --out result/baselines/hipporag2.friends.newname.t0
```

Baselines first build (commit) their memory session by session, saving a snapshot to
`run.snapshot_dir`, then answer the questions. Always pass a distinct `--out` per run:
the default (`./result/simulate_result.json`) is shared, so separate runs would overwrite
each other. The Oracle configs (`configs/baselines/oracle*.yaml`) run the same way.

**Outputs.** GroupChat-Mem writes a JSON per grid point to `run.result_dir`
(e.g. `result/groupchat_mem/friends/t0/`); a baseline writes `<--out>.<llm>.json`. Each
file holds a `summary` (`accuracy`, `n_scored`) and per-question `records`. Re-running the
same command resumes an interrupted run.

* * *

## Components

### Repository structure

```
.
├── README.md
├── requirements.txt
├── setup.cfg
├── .gitignore
├── configs/
│   ├── groupchat_mem/    # the proposed method
│   └── baselines/        # a_mem, mem0, hipporag2, oracle, simple_rag / direct_qa
├── prompts/              # prompt templates (OpenIE, QA, traversal, baselines)
├── preprocess/           # raw data -> experiment inputs (steps 1-5, Section 3)
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
| **Mem0** | Fact extraction with add/update/delete consolidation; hybrid retrieval (embedding cosine + BM25 scores). |
| **A-MEM** | Interlinked memory notes with dynamic linking and evolution. |
| **Oracle** | Upper bound that injects the answer sessions' evidence directly: Sessions (`oracle_sessions_only.*`), Records (`oracle_records_only.*`, ground-truth records), or Sessions + Records (`oracle.*`). |

Question format is selectable per config via `question_format`
(`multi_structured` / `multi_unstructured` / `open_ended`).

### Prompts

All prompt templates live in `prompts/`. Each is loaded by name (overridable per config
via the matching `prompt_*` key). QA prompts come in three question-format variants:
`*.txt` (multi_structured), `*.multi-unstructured.txt`, `*.open-ended.txt`.

All prompts are released **verbatim** as used in the reported experiments, including minor
wording slips and typos (some inherited from the upstream HippoRAG 2 / Mem0 prompts), so
that results can be reproduced exactly; editing a prompt changes the model's behavior.

| Prompt(s) | Used by | Purpose |
|---|---|---|
| `a_mem_note_construction.txt` | A-MEM | Build a memory note from a session (commit). |
| `a_mem_evolution.txt` | A-MEM | Link a new note to its neighbors and update their notes (single combined call). |
| `a_mem_query_keywords.txt` | A-MEM | Turn the question into retrieval keywords before searching notes. |
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
| `record_openie_extraction.txt` | GroupChat-Mem | OpenIE extraction that builds the relational memory $\mathcal{R}$ — turns each (partial) session into `[subject, relation, object]` records (precomputed before QA; each question only reads records available at its timepoint). |
| `query_node_id.txt` | GroupChat-Mem | Identify up to k query-relevant (subject) nodes of the question among the record-graph candidates. |
| `query_node_id_select.txt` | GroupChat-Mem | Select the top-k query-relevant nodes. |
| `qa_traversal_lookup.txt` | GroupChat-Mem | During iterative traversal, decide the next entity to look up. |

* * *

## Results

Main results on the New Names setting (mean over three runs), reported by the paper across
three backbones (Qwen3.5-35B-A3B, Gemma4-26B-A4B, GPT-OSS 20B); the best memory system per
column is in **bold** (Oracle rows are reference upper bounds and are not ranked). Each cell
shows accuracy (%) as Qwen3.5 / Gemma4 / GPT-OSS. The Oracle rows inject the answer
sessions' dialogue and/or their author-verified records (called Sessions+TKGs, Sessions
and TKGs in the paper).

| Memory System | Friends | The Big Bang Theory | The Office |
|---|---|---|---|
| Direct QA (No History) | 21.4 / 28.7 / 24.6 | 22.5 / 27.3 / 29.6 | 28.2 / 32.3 / 30.3 |
| Oracle: Sessions + Records (upper bound) | 96.7 / 96.8 / 93.2 | 95.5 / 95.1 / 92.8 | 91.8 / 92.9 / 91.0 |
| Oracle: Sessions | 77.2 / 79.2 / 77.5 | 85.1 / 82.0 / 85.8 | 77.0 / 78.0 / 78.0 |
| Oracle: Records | 64.1 / 64.2 / 63.1 | 63.7 / 65.3 / 64.6 | 58.2 / 59.8 / 60.1 |
| Simple RAG (BM25) | 69.9 / 69.3 / 56.2 | 69.7 / 68.9 / 61.5 | 65.8 / 64.5 / 59.6 |
| Simple RAG (OpenAI Embedding) | 70.4 / 70.2 / 58.7 | 74.9 / 76.0 / 67.8 | 74.3 / 74.6 / 67.0 |
| HippoRAG 2 | 73.2 / 73.0 / 60.1 | 78.1 / 77.4 / 68.1 | 75.1 / 75.3 / 68.5 |
| Mem0 | 71.2 / 71.5 / 62.3 | 73.9 / 71.4 / 64.9 | 68.8 / 69.9 / 66.0 |
| A-MEM | 65.2 / 69.2 / 58.0 | 69.6 / 70.2 / 61.8 | 68.1 / 69.3 / 62.7 |
| **GroupChat-Mem** | **77.8 / 77.6 / 64.8** | 77.5 / **78.9 / 70.3** | **78.7** / **77.1** / **69.0** |
