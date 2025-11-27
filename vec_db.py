from typing import Dict, List, Annotated
import numpy as np
import os
import json

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        num_records = self._get_num_records()
        centroids = np.load("centroids.npy")
        with open("ivf_lists.json", "r") as f:
            inverted_lists = json.load(f)
        inverted_lists = {int(k): v for k, v in inverted_lists.items()}
        dists = []
        for c in centroids:
            dists.append(1 - self._cal_score(c, query))
        dists = np.array(dists)
        nprobe = 3
        closest = np.argsort(dists)[:nprobe] # those are the closest nprobs clusters
        for cid in closest:
            for row_num in inverted_lists[cid]:
                vector = self.get_one_row(row_num)
                score = self._cal_score(query, vector)
                scores.append((score, row_num))

        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def kmeans(self , X, k, max_iters=100):
        """
        X: data points, shape (n_samples, n_features)
        k: number of clusters
        max_iters: maximum number of iterations
        """
        
        # 1. Randomly initialize cluster centroids
        #np.random.seed(42)
        random_indices = np.random.choice(len(X), k, replace=False)
        centroids = X[random_indices]

        for _ in range(max_iters):
            # 2. Assign points to closest centroid
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Compute new centroids from mean of points
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

            # 4. Stop if converged (no change)
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids

        return centroids, labels


    #clear cash each time get this python line    
    def _build_index(self):
        # Placeholder for index building logic
        rows = self.get_all_rows()
        n_clusters_testing = 2
        n_clusters = round(np.sqrt(len(rows))) # just for testing
        n_probes = round(np.sqrt(len(rows)//n_clusters)) # intial is 66
        max_iters = 10
        final_centroids, labels = VecDB.kmeans(rows, n_clusters,max_iters)
        inverted_lists = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(labels):
            inverted_lists[cluster_id].append(idx)
        np.save("centroids.npy", final_centroids)
        with open("ivf_lists.json", "w") as f:
            json.dump(inverted_lists, f)
        return final_centroids, inverted_lists