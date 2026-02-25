import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import time
import os


class DenseRetriever:
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2', sample_size=50000,
                 index_save_path="/data/faiss_index.bin",
                 df_save_path="/data/faiss_products.parquet"):
        """
        GPU-Accelerated Vector Retriever using FAISS with Disk Caching.
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.index_save_path = index_save_path
        self.df_save_path = df_save_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on device: {self.device}")

        # Load Model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.product_data = None

        self._load_or_build_index()

    def _load_or_build_index(self):
        # 1. Check if cache exists
        if os.path.exists(self.index_save_path) and os.path.exists(self.df_save_path):
            print(
                f"Found cached index. Loading from {self.index_save_path}...")

            # Load the exact mapping dataframe
            self.product_data = pd.read_parquet(self.df_save_path)

            # Load FAISS index (loads to CPU first)
            cpu_index = faiss.read_index(self.index_save_path)

            # Move to GPU for fast search
            if self.device == 'cuda':
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index

            print(
                f"Successfully loaded index with {self.index.ntotal} vectors.")

        else:
            # 2. Build from scratch if no cache
            print("No cache found. Building index from scratch...")
            print(f"Loading data from {self.data_path}...")
            df = pd.read_parquet(self.data_path)

            unique_products = df.drop_duplicates(subset=['product_id']).copy()
            if len(unique_products) > self.sample_size:
                unique_products = unique_products.sample(
                    n=self.sample_size, random_state=42)

            self.product_data = unique_products.reset_index(drop=True)

            print(
                f"Encoding {len(self.product_data)} products... (This uses the GPU)")
            corpus_embeddings = self.model.encode(
                self.product_data['text_corpus'].tolist(),
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            print("Building FAISS Index...")
            d = corpus_embeddings.shape[1]
            cpu_index = faiss.IndexFlatIP(d)
            cpu_index.add(corpus_embeddings)

            # Save to disk BEFORE moving to GPU (FAISS handles CPU index writing better)
            print(f"Saving FAISS index to {self.index_save_path}...")
            faiss.write_index(cpu_index, self.index_save_path)

            print(f"Saving product mapping to {self.df_save_path}...")
            self.product_data.to_parquet(self.df_save_path)

            # Move to GPU for active use
            if self.device == 'cuda':
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index

            print(f"Index Ready. Total vectors: {self.index.ntotal}")

    def search(self, query, top_k=10):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 if it runs out of matches
            product = self.product_data.iloc[idx]
            results.append({
                'product_id': product['product_id'],
                'title': product['product_title'],
                'score': float(distances[0][i]),
                'text': product['text_corpus']
            })

        return results

# --- TEST IT ---
# if __name__ == "__main__":
#     FINAL_DATA_PATH = ""
#     dense_retriever = DenseRetriever(FINAL_DATA_PATH)
#     print("\n--- BM25 FAILED HERE. HOW DOES VECTOR SEARCH DO? ---")
#     results = dense_retriever.search("running shoes for men", top_k=5)
#     for r in results:
#         print(f"[{r['score']:.4f}] {r['title']}")
