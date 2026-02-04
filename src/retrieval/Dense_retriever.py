import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import time


class DenseRetriever:
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2', sample_size=50000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Running on Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.product_data = None

        self._build_index()

    def _build_index(self):
        print(f"Loading data from the path:{self.data_path}")
        start_time = time.time()
        start_time = time.time()
        df = pd.read_parquet(self.data_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time taken to Load the data:{execution_time}")

        unique_products = df.drop_duplicates(subset=['product_id']).copy()

        len_unique_products = len(unique_products)

        if len_unique_products > self.sample_size:
            print(
                f"Downsampling from {len_unique_products} to {self.sample_size}")
            unique_products = unique_products.sample(
                n=self.sample_size, random_state=42)
        self.product_data = unique_products.reset_index(drop=True)

        # Generating the embeddings:

        print(f"Encoding {len(self.product_data)} products...")

        # we encode the text_corpus we created earlier
        corpus_embeddings = self.model.encode(
            self.product_data['text_corpus'],
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build the faiss index
        print("Building the faiss index...")

        d = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)

        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(corpus_embeddings)
        print(f"Index Ready. Total vectors : {self.index.ntotal}")

    def search(self, query, top_k=10):
        # 1. Encode Query
        query_vector = self.model.encode([query], convert_to_numpy=True)

        # 2. FAISS Search
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            product = self.product_data.iloc[idx]
            results.append({
                'product_id': product['product_id'],
                'title': product['product_title'],
                'score': float(distances[0][i]),  # Cosine Similarity score
                'text': product['text_corpus']
            })

        return results


# --- TEST IT ---
# if __name__ == "__main__":
#     dense_retriever = DenseRetriever()

#     print("\n--- BM25 FAILED HERE. HOW DOES VECTOR SEARCH DO? ---")
#     results = dense_retriever.search("running shoes for men", top_k=5)
#     for r in results:
#         print(f"[{r['score']:.4f}] {r['title']}")
