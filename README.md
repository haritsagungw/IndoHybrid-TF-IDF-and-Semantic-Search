# IndoHybrid-TF-IDF-and-Semantic-Search

**IndoHybrid** is a hybrid search engine designed for Indonesian text retrieval. It combines lexical and semantic search. The search results are merged using **Reciprocal Rank Fusion (RRF)** to deliver accurate Information Retrieval (IR) performance.

## Key Features

1. Lexical Search (Sparse Retrieval): Built using `scikit-learn` TF-IDF Vectorizer integrated with NLTK Indonesian stopword filtering and text preprocessing.
2. Semantic Search (Dense Retrieval): Utilizes `intfloat/multilingual-e5-small` embeddings paired with **FAISS (IndexFlatIP)** for fast vector indexing and inner product similarity search.
3. Rank Fusion: Combines sparse and dense retrieval rankings using **Reciprocal Rank Fusion (RRF)** to balance exact keyword matching and deep contextual understanding.
4. Persistence & Serialization: Includes built-in mechanisms to export and reload model artifacts (FAISS index, TF-IDF matrix, vectorizer, and metadata) for inference or deployment.
5. Benchmarking & Evaluation: Built-in evaluation pipeline measuring **Hit Rate** and **Mean Reciprocal Rank (MRR)**.

## Evaluation & Benchmarking

Evaluated on an Indonesian dataset (`andreaschandra/tydiqa-id`) using Top-10 Retrieval:

| Method | Hit Rate | MRR |
| :--- | :---: | :---: |
| **Lexical (TF-IDF)** | 84.00% | 0.6199 |
| **Semantic (E5 + FAISS)** | 90.00% | 0.6947 |
| **Hybrid (RRF)** | **88.00%** | **0.6865** |

## Installation & Setup

1. Clone this repository:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
```python
import pandas as pd
from IndoHybrid_TF_IDF_and_Semantic_Search import HybridSearch, Config

# Initialize configuration and hybrid search engine
config = Config()
searcher = HybridSearch(config)

# Prepare sample dataset
df = pd.DataFrame([
    {"id": "DOC_001", "title": "Kota Palembang", "text": "Kota Palembang adalah ibu kota provinsi Sumatera Selatan yang terkenal dengan Pempek."},
    {"id": "DOC_002", "title": "Seni Wayang", "text": "Wayang Kulit Palembang adalah seni pewayangan khas Sumatera Selatan."}
])

# Build indexes and perform query
searcher.build(df)
results = searcher.search("Informasi tentang Palembang", top_k=2)
print(results)
```

## Evaluation and Results

The model achieves an **88.00% Hit Rate@10** and **0.6865 MRR@10** on the Indonesian subset of the `andreaschandra/tydiqa-id` dataset benchmark.

| Query Input | Top Retrieved Document (Title) | Match Type | RRF Score |
| :--- | :--- | :--- | :---: |
| `makanan khas palembang` | Pempek Palembang | Lexical + Semantic | 0.0328 |
| `tari tradisional sumatra selatan` | Tari Tanggai | Semantic | 0.0164 |
| `ibu kota sumsel` | Kota Palembang | Semantic (Synonym) | 0.0164 |
   


   
