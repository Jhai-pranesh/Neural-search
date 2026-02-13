from reranker.Hybrid import HybridRetriever
import random
import tqdm

def mine_hard_negatives(retriever, df_data, num_queries=2000):
    """
    Mine triplets (Query, Positive, Negative) where the Negative 
    is a 'Hard Negative' (high rank in retrieval, low label in truth).
    """
    print("Generating hard negatives from queries...")
    query_groups = df_data.groupby('query')  # group the data by query
    all_queries = list(query_groups.groups.keys())

    # Shuffle and pick subset
    selected_queries = random.sample(
        all_queries, min(num_queries, len(all_queries)))

    training_triplets = []

    for query in tqdm(selected_queries):
        group = query_groups.get_group(query)

        # 1. Get positives (Exact matches only)
        positives = group[group['relevance_score']
                          == 3]['product_id'].to_list()
        # Skip if no exact matches exists
        if not positives:
            continue
        # run Retrieval (Simulate the mistake)
        # We look at Top 20 to find teh high-ranking mistakes
        results = retriever.search(query, top_k=20)

        hard_negatives = []

        for r in results:
            pid = r['product_id']
            # Check groud truth for this product
            # If it is in the group data AND has score 0 (Irrelevant)
