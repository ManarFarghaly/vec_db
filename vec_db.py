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
        query_1d = query.reshape(-1) 
        
        centroids = np.load("centroids.npy")
        with open("ivf_lists.json", "r") as f:
            inverted_lists = json.load(f)
        inverted_lists = {int(k): [int(x) for x in v] for k, v in inverted_lists.items()}
        
        # Euclidean Distance (L2 norm) for finding closest centroids
        distances = np.linalg.norm(centroids - query_1d, axis=1) 

        nprobe = 3 # Number of clusters to search
        # Find the indices of the nprobe closest clusters (smallest distance)
        closest_cluster_ids = np.argsort(distances)[:nprobe] 
        
        scores = []
        
        # Loop through the rows in the chosen inverted lists
        for cid in closest_cluster_ids:
            # Check if the cluster has any vectors
            if cid not in inverted_lists:
                continue

            for row_num in inverted_lists[cid]:
                # Retrieve the actual vector from the database file
                vector = self.get_one_row(row_num)
                if isinstance(vector, str): # Handle error case from get_one_row
                    print(f"Skipping row {row_num} due to error: {vector}")
                    continue
                    
                # Calculate the final similarity score (Cosine Similarity)
                score = self._cal_score(query_1d, vector)
                scores.append((score, row_num))

        # Sort by score (descending) then by row_num (ascending) for tie-breaking
        # Python's sort is stable, but we can be explicit
        scores.sort(key=lambda x: (-x[0], x[1]))
        
        # Return the top_k row indices
        return [s[1] for s in scores[:top_k]]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def kmeans(self , X, k, max_iters=100, batch_size=10000):
        """
        X: data points, shape (n_samples, n_features)
        k: number of clusters
        max_iters: maximum number of iterations
        """
        
        # 1. Randomly initialize cluster centroids
        #np.random.seed(42)
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int) # Array to store all labels
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_indices].copy()
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iters):
            # 2. Assign points to closest centroid (using batching)
            # We iterate over the data in batches to calculate and store the labels
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                
                # Calculate distances for the batch. The intermediate array is now:
                # (batch_size, k, dimension) which is much smaller.
                distances_batch = np.linalg.norm(X_batch[:, None] - centroids[None, :], axis=2)
                
                # Store the cluster IDs for this batch
                labels[i:i + batch_size] = np.argmin(distances_batch, axis=1)

            # 3. Compute new centroids from mean of points
            new_centroids = np.zeros_like(centroids)
           
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i] 

            # 4. Stop if converged (no change)
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids

        return centroids, labels


    #clear cash each time get this python line    
    def _build_index(self):
        # Placeholder for index building logic
        rows = self.get_all_rows()
        N = len(rows)
        if N == 0:
            print("Database is empty, skipping index build.")
            return None, None

        # Determine the number of clusters based on the number of records
        n_clusters = 1000
        # Ensure n_clusters is at least 1, and not more than N
        n_clusters = max(1, min(n_clusters, N))
        
        max_iters = 10
        final_centroids, labels = self.kmeans(rows, n_clusters, max_iters=max_iters)
        
        inverted_lists = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(labels):
            inverted_lists[cluster_id].append(idx)
            
        np.save("centroids.npy", final_centroids)
        with open("ivf_lists.json", "w") as f:
            json.dump(inverted_lists, f)
            
        # The return value should also reflect the actual computed cluster count
        return final_centroids, inverted_lists