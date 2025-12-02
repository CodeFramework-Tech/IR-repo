<<<<<<< HEAD
Hybrid information-retrieval pipeline that ingests the provided `data/news.csv`, builds multiple indexes locally (Boolean, TF‑IDF, BM25 + lightweight reranker), and exposes a CLI for building indexes and executing queries.

## Prerequisites

- Windows/macOS/Linux with Python 3.11+ (tested on 3.11.9 in `.venv311`; 3.13 system interpreter also available)
- Disk space for generated artifacts in `index/` (~20 MB for the supplied dataset)
- Optional: PowerShell on Windows for the activation commands below

## Quick Start

```powershell
# 1) Clone or download the repo, then inside project root:
python -m venv .venv311              # create venv (optional but recommended)
 source .venv311/bin/activate
.\.venv311\Scripts\Activate.ps1      # PowerShell; use activate.bat on cmd/activate on bash

python -m pip install --upgrade pip  # once per venv
python -m pip install -r requirements.txt

# 2) Build indexes (writes to index/)
python -m src.hybrid_search build

# 3) Run queries
python -m src.hybrid_search "covid pakistan"
```

> **Note:** Running from within `src/` also works (`python hybrid_search.py build`) if you prefer script paths, but the module form keeps imports consistent.

## Virtual Environment Tips

- Reuse the prepared environment with `.\.venv311\Scripts\Activate.ps1` (PowerShell) every time you open a new terminal.
- To leave the venv: `deactivate`.
- If you created a different interpreter version (e.g., system `python` 3.13), adjust the commands accordingly.

## Data & Indexes

- Source data lives in `data/news.csv`. Replace this file to work with another corpus of similar schema.
- Generated assets:
  - `index/boolean.pkl`
  - `index/tfidf_vectorizer.pkl`
  - `index/tfidf_matrix.pkl`
  - `index/bm25.pkl`
  - `index/metadata.json`
- Delete `index/` or rerun the build command whenever the dataset or preprocessing changes.

## Available Components

- `src/preprocess.py`: tokenization, normalization, lemmatization, and stop-word removal (includes runtime NLTK downloads for `stopwords`, `wordnet`, `omw-1.4`).
- `src/boolean_index.py`: inverted index + AND retrieval helper.
- `src/tfidf_index.py`: builds the scikit-learn TF‑IDF matrix and exposes query scoring.
- `src/bm25_index.py`: constructs a Rank-BM25 index and exposes query scoring.
- `src/re_ranker.py`: simple linear reranker consuming BM25/TF‑IDF features and document/query heuristics.
- `src/hybrid_search.py`: CLI entry point orchestrating preprocess → candidate filtering → scoring → reranking.

## Running Evaluations

- `python -m src.hybrid_search "query here"` prints the top 10 doc IDs, rerank scores, and 300-character snippets.
- For regression testing, there are starter tests (e.g., `src/test_hybrid.py`, `src/test_embeddings.py`) that can be executed inside the venv:
  ```powershell
  python -m pytest src/test_hybrid.py
  ```
  (Install `pytest` if needed.)

## Troubleshooting

- **`ModuleNotFoundError`:** ensure you are running commands from the project root _after_ activating the venv so `src/` is on `PYTHONPATH`.
- **`nltk` downloader prompts:** first run downloads required corpora into `%APPDATA%\nltk_data`. Allow the download; subsequent runs use the cached files.
- **Unicode errors on Windows consoles:** `hybrid_search.py` forces UTF‑8 output, but if you still see mojibake, switch to Windows Terminal or run `chcp 65001`.

## Project Structure

```
IR-PROJECT/
├── data/
│   └── news.csv
├── index/                # generated artifacts
├── src/
│   ├── hybrid_search.py
│   ├── preprocess.py
│   ├── boolean_index.py
│   ├── tfidf_index.py
│   ├── bm25_index.py
│   ├── re_ranker.py
│   └── ...
├── requirements.txt
├── assignment.md         # homework prompt
└── README.md             # this file
```
>>>>>>> e72b64a (added documentation)
