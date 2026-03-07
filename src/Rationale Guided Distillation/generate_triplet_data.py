from reranker.Hybrid import HybridRetriever
import random
import tqdm
import pandas as pd


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
            match = group[group['product_id'] == pid]

            if not match.empty:
                score = match.iloc[0]['relevance_score']
                if score <= 1:
                    hard_negatives.append(pid)
        # 3. Create Triplets
        # For every positive, pair it with a hard negative found
        if hard_negatives:
            # Pick one positive and one hard negative randomly to balance
            pos = random.choice(positives)
            neg = random.choice(hard_negatives)

            # Get text for them
            pos_text = group[group['product_id'] == pos].iloc[0]['text_corpus']
            neg_text = group[group['product_id'] == neg].iloc[0]['text_corpus']

            training_triplets.append({
                'query': query,
                'positive_id': pos,
                'positive_text': pos_text,
                'negative_id': neg,
                'negative_text': neg_text
            })

    print(f"Mined {len(training_triplets)} Hard Negative Triplets.")
    return pd.DataFrame(training_triplets)


df_clean = "Path_to_clean_data"

hybrid_retriever = HybridRetriever()
df_triplets = mine_hard_negatives(hybrid_retriever, df_clean, num_queries=2000)

# Save for the next step
# change the path if required
df_triplets.to_parquet("../data/hard_negatives_train.parquet")
