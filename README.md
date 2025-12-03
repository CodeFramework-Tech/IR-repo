# IR-Repo

## Introduction

This project implements an information retrieval (IR) system with a hybrid search mechanism. It allows for the generation of embeddings, indexing, and querying of data using a state-of-the-art IR framework. The system supports the evaluation of search performance and testing of hybrid search results.

## Features

- Hybrid search mechanism.
- Index creation and embedding generation.
- Search engine evaluation tools.
- Testable system for search and ranking.

## Requirements

- Python 3.x
- pip (for installing dependencies)
- Virtual Environment (optional but recommended)
- Any additional software or libraries required for indexing or embeddings.

## Troubleshooting

1. - Make sure you are using Python 3.x. Check the version by running `python --version`.

2. - Ensure that you have activated the virtual environment before running `pip install -r requirements.txt`.

3. - Check if the index was built correctly. Rebuild the index using `python -m src.build_indexes`.

## Example Usage

To run a search query:

```bash
python -m src.hybrid_search "Example query"
```

Set up virtual environment:

```bash
python3 -m venv .venv311
source .venv311/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/CodeFramework-Tech/IR-repo.git
cd IR-repo
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

Running the Project:

```bash
cd src
python -m src.build_indexes
python src/embeddings.py
python -m src.hybrid_search "your query here"
python -m src.evaluation
python -m src.test_hybrid
python -m src.hybrid_search "your query here"
```

Important info:
This project has been tested on macOS. Ensure you are using the correct Python version (3.x) and that dependencies like BERT and FAISS are installed correctly. Otherwise, you may encounter errors and the search engine might not function properly.

## Environment Setup

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```
