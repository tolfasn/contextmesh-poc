# core/vector_store_manager.py
# This module will manage the FAISS vector store.
import faiss # type: ignore
import numpy as np
import os

class FaissVectorStore:
    def __init__(self, index_file_path: str = 'contextmesh_poc.faiss', dimension: int = 384):
        """
        Initializes the FaissVectorStore.
        :param index_file_path: Path to save/load the FAISS index file.
        :param dimension: Dimension of the embeddings (e.g., 384 for bge-small-en-v1.5).
        """
        self.index_file_path = index_file_path
        self.dimension = dimension
        self.index: faiss.Index | None = None # Main index, could be IndexFlatL2 or IndexIDMap
        self.doc_id_map: list[str] = [] # To map FAISS internal IDs back to original document identifiers

        self.load_index() # Attempt to load existing index

    def add_embeddings(self, embeddings: np.ndarray, doc_identifiers: list[str]):
        """
        Adds embeddings to the FAISS index along with their document identifiers.
        :param embeddings: A NumPy array of embeddings.
        :param doc_identifiers: A list of unique string identifiers for each document/text.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension}).")
        if len(embeddings) != len(doc_identifiers):
            raise ValueError("Number of embeddings and document identifiers must be the same.")

        if self.index is None:
            # Using IndexFlatL2 as it's simple and good for PoC.
            # For larger datasets, more advanced indexing like IndexIVFFlat might be considered.
            self.index = faiss.IndexFlatL2(self.dimension)
            print(f"Initialized new FAISS IndexFlatL2 with dimension {self.dimension}.")

        # Add embeddings to FAISS
        self.index.add(embeddings.astype(np.float32)) # FAISS expects float32

        # Store document identifiers corresponding to the order they were added
        # FAISS internal IDs are sequential (0, 1, 2...) for IndexFlatL2
        current_offset = len(self.doc_id_map)
        for i, doc_id in enumerate(doc_identifiers):
            # We could also store a mapping from FAISS ID (current_offset + i) to doc_id
            self.doc_id_map.append(doc_id)

        print(f"Added {len(embeddings)} embeddings. Index total: {self.index.ntotal}. Doc ID map size: {len(self.doc_id_map)}")


    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        Searches for k most similar embeddings.
        :param query_embedding: A single query embedding (1D NumPy array).
        :param k: Number of similar items to return.
        :return: A list of tuples (doc_identifier, distance_score).
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is not initialized or is empty.")
            return []

        if query_embedding.ndim == 1:
            query_embedding_2d = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding_2d = query_embedding.astype(np.float32)
        else:
            raise ValueError("Query embedding must be a 1D array or a 2D array with one row.")

        if query_embedding_2d.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension ({query_embedding_2d.shape[1]}) does not match index dimension ({self.dimension}).")

        # Ensure k is not greater than the number of items in the index
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            return []

        distances, indices = self.index.search(query_embedding_2d, actual_k)

        results = []
        for i in range(actual_k):
            faiss_id = indices[0][i]
            distance_score = distances[0][i]
            if 0 <= faiss_id < len(self.doc_id_map):
                doc_id = self.doc_id_map[faiss_id]
                results.append((doc_id, float(distance_score)))
            else:
                print(f"Warning: FAISS index {faiss_id} out of bounds for doc_id_map (size {len(self.doc_id_map)}).")
        return results

    def save_index(self):
        """Saves the FAISS index and document ID map to files."""
        if self.index:
            faiss.write_index(self.index, self.index_file_path)
            print(f"FAISS index saved to {self.index_file_path}")

            # Save the doc_id_map as well
            doc_id_map_file = self.index_file_path + ".map"
            with open(doc_id_map_file, 'w') as f:
                for doc_id in self.doc_id_map:
                    f.write(f"{doc_id}\n")
            print(f"Document ID map saved to {doc_id_map_file}")
        else:
            print("No index to save.")

    def load_index(self):
        """Loads the FAISS index and document ID map from files."""
        if os.path.exists(self.index_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                print(f"FAISS index loaded from {self.index_file_path}. Total entries: {self.index.ntotal}, Dimension: {self.index.d}")
                if self.index.d != self.dimension:
                    print(f"Warning: Loaded index dimension ({self.index.d}) differs from configured dimension ({self.dimension}). Resetting index.")
                    self.index = None # Invalidate index
                    self.doc_id_map = []
                    return


                doc_id_map_file = self.index_file_path + ".map"
                if os.path.exists(doc_id_map_file):
                    with open(doc_id_map_file, 'r') as f:
                        self.doc_id_map = [line.strip() for line in f.readlines()]
                    print(f"Document ID map loaded from {doc_id_map_file}. Size: {len(self.doc_id_map)}")

                    if self.index.ntotal != len(self.doc_id_map):
                        print(f"Warning: Mismatch between FAISS index size ({self.index.ntotal}) and doc_id_map size ({len(self.doc_id_map)}). Index may be corrupt or map is outdated. Resetting.")
                        self.index = None
                        self.doc_id_map = []
                else:
                    print(f"Warning: Document ID map file {doc_id_map_file} not found. Index loaded but mapping will be lost if not rebuilt.")
                    # If map is missing, it's safer to assume the index is not usable without it
                    self.index = None
                    self.doc_id_map = []


            except Exception as e:
                print(f"Error loading FAISS index or map: {e}. A new index will be created if data is added.")
                self.index = None
                self.doc_id_map = []
        else:
            print(f"Index file {self.index_file_path} not found. A new index will be created on first add.")
            self.index = None
            self.doc_id_map = []


if __name__ == '__main__':
    # Example Usage
    try:
        # Ensure the embedding model and vector store have matching dimensions
        from embedding_generator import EmbeddingModel # Assumes embedding_generator.py is in the same directory or python path

        emb_model = EmbeddingModel() # This will print its dimension
        store_dimension = emb_model.dimension

        print(f"\n--- VectorStore Example (Dimension: {store_dimension}) ---")
        # Clean up previous test files
        test_index_file = "test_poc_index.faiss"
        if os.path.exists(test_index_file): os.remove(test_index_file)
        if os.path.exists(test_index_file + ".map"): os.remove(test_index_file + ".map")

        store = FaissVectorStore(index_file_path=test_index_file, dimension=store_dimension)

        sample_texts_for_store = [
            "Old commit about user authentication.",
            "Recent fix for login page.",
            "Documentation update for API."
        ]
        sample_doc_ids = ["commit_abc1", "commit_def2", "doc_xyz3"]

        print("\nGenerating embeddings for store...")
        test_embeddings = emb_model.generate_embeddings(sample_texts_for_store)

        if test_embeddings is not None:
            print("\nAdding embeddings to store...")
            store.add_embeddings(test_embeddings, sample_doc_ids)
            store.save_index()

            print("\nCreating new store instance and loading index...")
            new_store = FaissVectorStore(index_file_path=test_index_file, dimension=store_dimension)
            # new_store.load_index() # load_index is called in __init__

            if new_store.index is not None:
                print("\nGenerating query embedding...")
                query_text = "Information about user login"
                query_embedding = emb_model.generate_embeddings([query_text])

                if query_embedding is not None:
                    print(f"\nSearching for '{query_text}' (k=2)...")
                    results = new_store.search_similar(query_embedding[0], k=2)
                    print("Search results (doc_id, distance):")
                    for doc_id, dist in results:
                        print(f"  ID: {doc_id}, Distance: {dist:.4f}")
                else:
                    print("Failed to generate query embedding.")
            else:
                print("Failed to load index into new_store.")
        else:
            print("Failed to generate test embeddings for store.")

        # Clean up test files
        # if os.path.exists(test_index_file): os.remove(test_index_file)
        # if os.path.exists(test_index_file + ".map"): os.remove(test_index_file + ".map")

    except Exception as e:
        print(f"An error occurred in FaissVectorStore example: {e}")
        import traceback
        traceback.print_exc()