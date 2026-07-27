# Temporal Knowledge-Graph Retrieval for Long-Term Group-Chat Question Answering

This repository contains the code for studying **question answering over long-term, multi-party (group-chat) conversations**. Given a long dialogue history among several speakers, a model must answer time-sensitive questions (e.g. *"Which company did Daniel join?"*) that require retrieving and reasoning over information scattered across many past sessions. We formulate the task as memory + retrieval over an evolving conversation and evaluate on a DialSim-derived benchmark spanning three shows (Friends, BigBang, TheOffice). Our proposed method builds a **temporal knowledge graph** from the dialogue and performs seed-node-guided, iterative retrieval that combines graph facts with dialogue passages, avoiding the future-information leakage that naïve snapshot retrieval introduces. We compare against strong memory baselines (A-MEM, Mem0, HippoRAG2) and an oracle upper bound.

## Key Features

- **Temporal KG + retrieval method** (`tkg_hybrid_bridge_inverse`): extracts timestamped quadruples per session, identifies question "seed nodes", and iteratively expands over the graph with dialogue passage fallback. Retrieval is reconstructed at each query's timepoint so no future session leaks into a past question.
- **Multiple memory baselines** implemented under a common interface: **A-MEM**, **Mem0**, **HippoRAG2**, and an **Oracle** upper bound.
- **Three question formats** — multiple-choice structured, multiple-choice unstructured, and open-ended — selectable per config.
- **Local LLM serving** via an OpenAI-compatible API (e.g. vLLM), with load-aware request distribution across multiple endpoints.
- **Embedding cache** for dialogue / question / quadruple embeddings (OpenAI `text-embedding-3-small`) to avoid redundant API calls.
- **Reproducible experiment driver** with per-session checkpointing, resume, and grid ablations.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/                    # Example experiment configs (YAML)
├── prompts/                    # Prompt templates (extraction, QA, baselines)
└── src/
    ├── groupchat_qa/
    │   ├── client/             # LLM clients (OpenAI-compatible / vLLM), load-aware routing
    │   ├── embedder/           # Embedders + on-disk embedding cache
    │   ├── memory/             # Memory methods: proposed TKG method + A-MEM / Mem0 / HippoRAG2 / Oracle
    │   ├── pipeline/           # Data loading and QA pipeline
    │   └── utils/              # Tokenization, formatting, filesystem helpers
    └── run/
        ├── main.ablation_bridge_inverse.py   # Entry point for the proposed method
        ├── main.simulate_bridge_inverse.py   # Shared runner used by the proposed-method entry point
        └── main.simulate.py                  # Entry point for baselines + oracle (commit + QA)
```

## Installation

Requires **Python ≥ 3.10** (the code uses PEP 604 `X | Y` type unions).

```bash
git clone <REPO_URL>
cd <REPO_DIR>

python -m venv .venv        # Python >= 3.10
source .venv/bin/activate
pip install -r requirements.txt
```

Set the required environment variables (create a `.env` file or export directly):

```bash
export OPENAI_API_KEY=sk-...   # used for text-embedding-3-small embeddings
```

**Local LLM serving.** The generation models are served locally through an OpenAI-compatible HTTP API (e.g. [vLLM](https://github.com/vllm-project/vllm)). Each config points at one or more endpoints (host / port); start your server(s) and set the endpoints accordingly before running experiments. No proprietary API is required for generation.

## Data

Experiments use a **DialSim-derived** benchmark of long-term group-chat dialogues for three shows (Friends, BigBang, TheOffice). Each show is available in several `data_type` variants (e.g. `newname`, `normal`, `shuffled`) and multiple trials. Per session we store the dialogue, per-timepoint question sets, and precomputed quadruples/embeddings under a `data/` directory tree that the configs reference via `dataset_root`.

The raw benchmark data is **not** included in this repository. Point `dataset_root` in the configs at your local copy of the preprocessed data.

## Usage

Both entry points read a YAML config and take an LLM identifier. Replace endpoints in the config with your local server(s), then run:

**Proposed method (temporal KG retrieval):**

```bash
PYTHONPATH=src python src/run/main.ablation_bridge_inverse.py \
    --config configs/nodia/nodia.qwen.friends.newname.t0.yaml \
    --method tkg_hybrid_bridge_inverse \
    --llms qwen3.5-35b-a3b-on
```

`--method` must match the config's top-level key (`tkg_hybrid_bridge_inverse`). The
`configs/nodia/` directory also contains the `enhanced`, `tkg_only`, `rag_only`, and
`bridge_inverse` retrieval variants used in the comparison.

Use `--grid-start` / `--grid-end` to run a slice of the ablation grid, and `--rerun-missing` to re-process only sessions that were not yet scored.

**A baseline (e.g. Mem0 / A-MEM / HippoRAG2):**

```bash
PYTHONPATH=src python src/run/main.simulate.py \
    --config configs/baselines/mem0.qwen.friends.newname.t0.yaml \
    --llms qwen3.5-35b-a3b-on \
    --out result/mem0.json
```

Baseline configs live in `configs/baselines/` (`a_mem`, `mem0`, `hipporag2`, `oracle`).

`main.simulate.py` first builds (commits) the memory from the dialogue, then answers the questions; pass `--commit-only` to build the memory snapshot without running QA.

## Results

Results are produced per (dataset × method × trial) as JSON files containing per-question predictions and an accuracy summary. We report answer accuracy on the multiple-choice benchmark and analyze retrieval quality against the oracle upper bound.

<!-- TODO: add the results table once final numbers are ready. -->

| Method | Friends | BigBang | TheOffice |
|--------|---------|---------|-----------|
| A-MEM      | TODO | TODO | TODO |
| Mem0       | TODO | TODO | TODO |
| HippoRAG2  | TODO | TODO | TODO |
| **Ours (TKG retrieval)** | TODO | TODO | TODO |
| Oracle (upper bound) | TODO | TODO | TODO |

## Citation

```bibtex
@article{TODO_citation_key,
  title   = {TODO: Paper Title},
  author  = {TODO: Authors},
  journal = {TODO: Venue},
  year    = {TODO}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
