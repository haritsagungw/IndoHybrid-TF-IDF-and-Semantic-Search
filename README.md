# IndoHybrid-TF-IDF-and-Semantic-Search

**IndoHybrid** is a hybrid search engine designed for Indonesian text retrieval. It combines lexical search (sparse retrieval via TF-IDF) and semantic search (dense retrieval via `intfloat/multilingual-e5-small` & FAISS). The search results are merged using **Reciprocal Rank Fusion (RRF)** to deliver accurate Information Retrieval (IR) performance.

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

---

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

   # a. Initialize Configuration and Hybrid Search Engine
   config = Config()
   searcher = HybridSearch(config)

   # b. Prepare Sample Data
   data = {
    "id": ["DOC_001", "DOC_002"],
    "title": ["Palembang City", "Wayang Art"],
    "text": [
        "Kota Palembang adalah ibu kota provinsi Sumatera Selatan.",
        "Wayang Kulit Palembang adalah seni pewayangan khas Sumatera Selatan."
    ]
   }
   df = pd.DataFrame(data)

   # c. Build Indexes
   searcher.build(df)

   # d. Perform Search Query
   results = searcher.search("Informasi tentang Palembang", top_k=2)

   for res in results:
    print(f"[{res['rank']}] {res['title']} | RRF Score: {res['rrf_score']}")
   ```

   ## Repository Structure
   .
├── IndoHybrid_TF_IDF_and_Semantic_Search.ipynb  # Main Jupyter Notebook
├── requirements.txt                              # Python dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
   


   
