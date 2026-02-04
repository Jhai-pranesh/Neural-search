from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever


class HybridRetriever:
    """
    I implemented a hybrid retrieval method using both BM25 (lexical search) and dense vector search (FAISS).
    I over-fetch results from both systems, convert their rankings into rank maps, and use Reciprocal Rank Fusion (RRF) to score all candidates retrieved by either system.
    Products that rank highly in both systems are naturally boosted, while those appearing in only one system are penalized but not discarded.
    Finally, I sort by the fused RRF score and return the top results.
    """

    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def search(self, query, top_k=50, k=60):
        """Performs Hybrid Search using RRF (Reciprocal Rank Fusion)
        args:
            top_k : Number of final results to return
            k : RRF constant(default = 60)"""

        # 1. We get results from both the previous methods to ensure overlap (that's why there is top_k * 2)and we can then apply the next steps
        bm25_results = self.bm25.search(query, top_k=k*2)
        dense_results = self.dense.search(query, top_k=k*2)

        # 2. Then create Rank maps
        # Map product_id -> rank(0-based)
        bm25_ranks = {r['product_id']: i for i, r in enumerate(bm25_results)}
        dense_ranks = {r['product_id']: i for i, r in enumerate(dense_results)}

        # 3. calculate the RRF scores
        # we consider all unique products found by both systems means that
        # Give me every product that was retrieved by BM25 OR dense search (or both).
        # all_ids = BM25 ∪ Dense
        all_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        hybrid_scores = []

        for pid in all_ids:
            # Get the ranks of all if not found make them a large number(like float('inf')), we are penalizing if the product is not there because in the next line when we calculate the score for RRF we will get 1 / (k + ∞) → 0 and the system contributes to nothing
            rank_bm25 = bm25_ranks.get(pid, float('inf'))
            rank_dense = dense_ranks.get(pid, float('inf'))

            # RRF formula
            score = (1/(k+rank_bm25)) + (1/(k+rank_dense))

            if pid in dense_ranks:
                meta = dense_results[dense_ranks[pid]]
            else:
                meta = bm25_results[bm25_ranks[pid]]

            hybrid_scores.append({
                'product_id': pid,
                'title': meta['title'],
                'score': score,
                'text': meta['text']
            })

        hybrid_scores = sorted(
            hybrid_scores, key=lambda x: x['score'], reverse=True)

        return hybrid_scores[:top_k]


FINAL_DATA_PATH = ""
bm25 = BM25Retriever(FINAL_DATA_PATH)
dense = DenseRetriever(FINAL_DATA_PATH)
hybrid_retriever = HybridRetriever(bm25, dense)
print("\n--- HYBRID SEARCH RESULTS (RRF) ---")
results = hybrid_retriever.search("running shoes for men", top_k=5)
for r in results:
    print(f"[{r['score']:.4f}] {r['title']}")
