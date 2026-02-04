import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import string
from tqdm import tqdm
import gc


class BM25Retriever:
    def __init__(self, data_path, sample_size=50000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.bm25 = None
        self.product_data = None

        self._build_index()

    def _preprocess_text(self, text):
        if text is None:
            return []
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def _build_index(self):
        print(f'Loading from the path: {self.data_path} ...')
        try:
            df = pd.read_parquet(self.data_path)
            unique_products = df.drop_duplicates(subset=['product_id']).copy()

            # delete df for minimizing the memory consumption
            del df
            gc.collect()

            len_unique_products = len(unique_products)

            # Downsampling of data if the data is too high
            if len_unique_products > self.sample_size:
                print(
                    f"Downsampling from {len_unique_products} to {self.sample_size} ...")
                unique_products = unique_products.sample(
                    n=self.sample_size, random_state=42)
            self.product_data = unique_products.reset_index(drop=True)

            # delete the unique_products for better memory capacity
            del unique_products
            gc.collect()

            # Tokenizing the text corpus
            print("Tokenizing the text corpus...")
            corpus_tokens = [
                self._preprocess_text(text) for text in tqdm(self.product_data['text_corpus'])]
            print("Building BM25 Index...")
            self.bm25 = BM25Okapi(corpus_tokens)
            print("BM25 Index Ready.")
        except MemoryError as e:
            print("Memory limit Exceeded...")
            print("Minimize the sample_size...")

            # collect the garbage
            gc.collect()

            # raise a runtime error due to insufficient memory to allocate
            raise RuntimeError(
                "BM25 index construction failed due to insufficient memory."
            ) from e

    def search(self, query, top_k=10):
        # tokenize the query
        tokenized_query = self._preprocess_text(query)

        # get the BM25 scores of the query
        scores = self.bm25.get_scores(tokenized_query)
        # until now the scores are not sorted

        """get the top k elements
        since the argsort returns in ascending order we are reversing the result and getting the top K elements
        top_n_indices = np.argsort(scores)[::-1][:top_k] -> it sorts the whole scores but when scaled it will cause latency hence we will use a optimized approach"""

        top_n_indices = np.argpartition(scores, -top_k)[-top_k:]
        # -> argpartion makes the array like this
        # [ smaller stuff | bigger stuff ] it partitions the array and we are sorting the last top_k since they will be the bigger ones but in unsorted order
        top_n_indices = top_n_indices[np.argsort(scores[top_n_indices])[::-1]]

        results = []
        for idx in top_n_indices:
            product = self.product_data.iloc[idx]
            results.append({
                'product_id': product['product_id'],
                'title': product['product_title'],
                'score': scores[idx],
                'text': product['text_corpus']
            })
        return results


# if __name__ == "__main__":
#     # Quick Test
#     # change the data path acoordingly
#     data_path = ""

#     retriever = BM25Retriever(data_path)
#     results = retriever.search("running shoes", top_k=3)

#     for r in results:
#         print(f"[{r['score']:.4f}] {r['title']}")
