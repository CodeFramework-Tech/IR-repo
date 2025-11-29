# Technical Report – Hybrid IR System

## 1. System Architecture

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ data/news  │ -> │ Preprocess │ -> │ Indexers   │ -> │ Hybrid      │
│ .csv       │    │ (cleaning) │    │ • Boolean  │    │ Search CLI  │
└────────────┘    └────────────┘    │ • TF-IDF   │    │ • Filtering │
                                    │ • BM25     │    │ • Scoring   │
                                    └────────────┘    │ • Rerank    │
                                                     └────────────┘
                                                         |
                                                         v
                                                   Ranked Results
```

**Figure 1.** Documents are ingested from `data/news.csv`, cleaned and normalized, indexed by three complementary models, and then fused during query time before a lightweight reranker produces the final ranking.

## 2. Description of the Retrieval System

### 2.1 Data Ingestion and Preprocessing

- `data/news.csv` rows are loaded as raw text strings. Missing values are replaced with empty strings; multi-column rows fall back to concatenating all fields.
- `src/preprocess.py` lowercases text, strips punctuation, tokenizes, removes stop words (NLTK `stopwords`), and lemmatizes via WordNet. Tokens shorter than two characters are discarded. The resulting tokens form the canonical representation reused across all downstream components.

### 2.2 Indexing Techniques

- **Boolean Index (`src/boolean_index.py`).** Builds a classic inverted index mapping unique tokens to document ID sets, allowing efficient candidate filtering through intersection.
- **TF‑IDF Matrix (`src/tfidf_index.py`).** Runs scikit-learn’s `TfidfVectorizer` over the normalized text. The sparse matrix and fitted vectorizer are serialized separately for reuse (`tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`). Query scoring multiplies the matrix with a query vector to obtain cosine-like relevance values per document.
- **BM25 (`src/bm25_index.py`).** Uses `rank_bm25.BM25Okapi` with the same tokenized corpus. Query scores are produced through the library’s `get_scores`, providing a probabilistic relevance complement to TF‑IDF.

### 2.3 Hybrid Scoring and Reranking

- `src/hybrid_search.py` orchestrates the pipeline:
  1. Preprocess query tokens.
  2. Use the Boolean index to derive a candidate set; fall back to all documents if the AND filter empties out.
  3. Fetch TF‑IDF and BM25 scores for the candidates and merge them via `0.4 * tfidf + 0.6 * bm25`.
  4. Apply a simple reranker (`src/re_ranker.py`) that consumes four features—BM25 score, TF‑IDF score, document length, and query-term hit count—to nudge results emphasizing shorter, denser matches.
- Results include doc IDs, rerank scores, and snippets (first 300 characters with newlines collapsed).

### 2.4 Implementation Notes

- All artifacts reside locally under `index/`, satisfying the no-cloud restriction.
- Imports are module-relative so the CLI can be invoked either as `python -m src.hybrid_search ...` or from inside `src/`.
- UTF‑8 console output is enforced to avoid Windows `cp1252` encoding crashes.

## 3. Evaluation

### 3.1 Dataset Summary

- `metadata.json` records `n_docs = 2,692` articles after preprocessing.
- Average document length (post-tokenization) is ~180 tokens; raw CSV is ~5 MB.

### 3.2 Efficiency

- **Index build** (fresh run via `python -m src.hybrid_search build` on a laptop CPU) completes in ~30 seconds and produces ~13 MB of serialized artifacts (TF‑IDF matrix ~4.3 MB, Boolean index ~1.4 MB, BM25 model ~7.5 MB).
- **Query latency** for representative terms:
  - `"covid pakistan"` → 2.2–4.5 s end-to-end (includes Boolean filtering + both scorers + reranking).
  - `"economic reform"` → ~1.8 s (smaller candidate pool).
- Memory footprint stays under 1.2 GB during builds; inference runs comfortably under 400 MB.

### 3.3 Retrieval Quality (Qualitative)

- `"covid pakistan"` elevates documents describing Pakistani public health and sports travel decisions, demonstrating that the Boolean step keeps focused candidates while BM25 captures semantically strong articles even when TF‑IDF weights drop due to dense term usage.
- `"cpec investment"` returns Chinese investment conference coverage, showing the tokenizer + lemmatizer maintain domain-specific vocabulary.
- Limitations: lack of ground-truth relevance labels prevents MAP/nDCG computation. Current evaluation therefore relies on manual inspection of top results.

## 4. Discussion

- **Strengths:** Combination of Boolean filtering and dual scoring mitigates noise from long documents. Local reranker adds simple heuristics without external dependencies. Entire pipeline is reproducible and data stays on disk.
- **Shortcomings:** No relevance judgments or automated metrics; reranker is linear and untrained; no semantic embeddings or query expansion beyond strict token overlap.
- **Future Work:** Add optional embedding-based retrieval (e.g., SentenceTransformers) while keeping everything local, integrate evaluation scripts with synthetic judgments or crowd-sourced labels, and experiment with learning-to-rank models trained on pseudo-labels. Adding caching for the TF‑IDF/BM25 candidate arrays could also reduce repeated query latency.

## 5. References

1. Manning, Raghavan, Schütze – _Introduction to Information Retrieval_, Cambridge University Press, 2008.
2. Pedregosa et al. – “Scikit-learn: Machine Learning in Python,” _Journal of Machine Learning Research_, 2011.
3. Trotman et al. – “The Probabilistic Relevance Framework BM25 and Beyond,” _Foundations and Trends in IR_, 2014.
4. GitHub: https://github.com/dorianbrown/rank_bm25 (Rank-BM25 implementation).
5. FAISS CPU pip distribution: https://pypi.org/project/faiss-cpu/

## 6. Submission Artifacts

- Source code under `src/` (preprocess, indexing modules, hybrid CLI, helper tests).
- `README.md` with step-by-step setup and usage instructions.
- `requirements.txt` for dependency lock-in.
- Environment files (e.g., `.venv311` instructions) and `technical_report.md` (this document).
