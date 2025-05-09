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
        self.index: faiss.Index | None = None 
        self.doc_id_map: list[str] = [] 

        self.load_index() 

    def add_embeddings(self, embeddings: np.ndarray, doc_identifiers: list[str]):
        """
        Adds embeddings to the FAISS index along with their document identifiers.
        :param embeddings: A NumPy array of embeddings.
        :param doc_identifiers: A list of unique string identifiers for each document/text.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings must be a 2D array with shape (n_embeddings, {self.dimension}). Got {embeddings.shape}")
        if len(embeddings) != len(doc_identifiers):
            raise ValueError("Number of embeddings and document identifiers must be the same.")
        if not all(isinstance(doc_id, str) for doc_id in doc_identifiers):
            raise ValueError("All document identifiers must be strings.")


        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            print(f"Initialized new FAISS IndexFlatL2 with dimension {self.dimension}.")

        self.index.add(embeddings.astype(np.float32)) 

        self.doc_id_map.extend(doc_identifiers)

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

        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            return []

        distances, indices = self.index.search(query_embedding_2d, actual_k)

        results = []
        for i in range(actual_k):
            faiss_id = indices[0][i] # This is the internal sequential ID from FAISS
            distance_score = distances[0][i]
            if 0 <= faiss_id < len(self.doc_id_map):
                doc_id = self.doc_id_map[faiss_id]
                results.append((doc_id, float(distance_score)))
            else:
                print(f"Warning: FAISS index {faiss_id} out of bounds for doc_id_map (size {len(self.doc_id_map)}). This might happen if map and index are desynced.")
        return results

    def save_index(self):
        """Saves the FAISS index and document ID map to files."""
        if self.index and self.index.ntotal > 0 : # Only save if there's something to save
            faiss.write_index(self.index, self.index_file_path)
            print(f"FAISS index saved to {self.index_file_path} with {self.index.ntotal} entries.")

            doc_id_map_file = self.index_file_path + ".map"
            with open(doc_id_map_file, 'w') as f:
                for doc_id in self.doc_id_map:
                    f.write(f"{doc_id}\n")
            print(f"Document ID map saved to {doc_id_map_file} with {len(self.doc_id_map)} entries.")
        else:
            print("No index data to save (index is None or empty).")

    def load_index(self):
        """Loads the FAISS index and document ID map from files."""
        loaded_successfully = False
        if os.path.exists(self.index_file_path):
            doc_id_map_file = self.index_file_path + ".map"
            if not os.path.exists(doc_id_map_file):
                print(f"Warning: Index file {self.index_file_path} exists, but map file {doc_id_map_file} is missing. Cannot reliably load.")
                self.index = None
                self.doc_id_map = []
                return

            try:
                print(f"Attempting to load FAISS index from {self.index_file_path}...")
                self.index = faiss.read_index(self.index_file_path)
                print(f"FAISS index loaded. Total entries: {self.index.ntotal}, Dimension: {self.index.d}")

                if self.index.d != self.dimension:
                    print(f"Error: Loaded index dimension ({self.index.d}) differs from configured dimension ({self.dimension}). Index will not be used.")
                    self.index = None 
                    self.doc_id_map = []
                    return # Critical mismatch

                print(f"Attempting to load document ID map from {doc_id_map_file}...")
                with open(doc_id_map_file, 'r') as f:
                    self.doc_id_map = [line.strip() for line in f.readlines() if line.strip()] # Ensure no empty lines
                print(f"Document ID map loaded. Size: {len(self.doc_id_map)}")

                if self.index.ntotal != len(self.doc_id_map):
                    print(f"Error: Mismatch between FAISS index size ({self.index.ntotal}) and doc_id_map size ({len(self.doc_id_map)}). Index is corrupt or map is outdated. Index will not be used.")
                    self.index = None
                    self.doc_id_map = []
                    return # Critical mismatch

                loaded_successfully = True

            except Exception as e:
                print(f"Error loading FAISS index or map: {e}. A new index will be created if data is added.")
                self.index = None
                self.doc_id_map = []
        else:
            print(f"Index file {self.index_file_path} not found. A new index will be created on first add.")
            self.index = None # Ensure it's None if no file found
            self.doc_id_map = []

        if not loaded_successfully and self.index is None : # Ensure index is None if loading failed or no file
            print("Initializing with an empty index structure as no valid existing index was loaded.")
            # self.index = faiss.IndexFlatL2(self.dimension) # Don't initialize here, let add_embeddings do it.

if __name__ == '__main__':
    try:
        # Dynamically get dimension from EmbeddingModel
        from embedding_generator import EmbeddingModel 

        print("--- FaissVectorStore __main__ Test ---")
        # Initialize EmbeddingModel to get the correct dimension for our vector store
        # This will also trigger model download if it's the first time for this model.
        print("Initializing EmbeddingModel to determine vector dimension...")
        emb_model = EmbeddingModel() # Uses default 'BAAI/bge-small-en-v1.5'
        store_dimension = emb_model.dimension
        print(f"Using vector dimension: {store_dimension}")

        test_index_file = "test_poc_index.faiss" # Will create test_poc_index.faiss and test_poc_index.faiss.map

        # Clean up previous test files before starting
        print(f"\nCleaning up old test files: {test_index_file} and {test_index_file}.map (if they exist)...")
        if os.path.exists(test_index_file): os.remove(test_index_file)
        if os.path.exists(test_index_file + ".map"): os.remove(test_index_file + ".map")

        print("\n--- Test 1: Creating new store, adding embeddings, saving ---")
        store1 = FaissVectorStore(index_file_path=test_index_file, dimension=store_dimension)

        sample_texts_for_store = [
            "Commit message: Fixed critical login bug.", # id: commit_001
            "Documentation: Updated API for user endpoint.", # id: doc_alpha
            "Commit message: Refactored payment processing module.", # id: commit_002
            "Chat discussion: Decision to use PostgreSQL for new service." # id: chat_thread_123
        ]
        sample_doc_ids = ["commit_001", "doc_alpha", "commit_002", "chat_thread_123"]

        print("\nGenerating embeddings for initial store content...")
        initial_embeddings = emb_model.generate_embeddings(sample_texts_for_store)

        if initial_embeddings is not None:
            print("\nAdding embeddings to store1...")
            store1.add_embeddings(initial_embeddings, sample_doc_ids)
            store1.save_index() # Save after adding
        else:
            print("Failed to generate initial embeddings. Test cannot continue fully.")
            exit()

        print("\n--- Test 2: Creating another store instance and loading from disk ---")
        store2 = FaissVectorStore(index_file_path=test_index_file, dimension=store_dimension)
        # load_index() is called in __init__. Check if it loaded correctly.
        if store2.index is not None and store2.index.ntotal > 0:
            print(f"Store2 loaded successfully. Index has {store2.index.ntotal} entries.")
            if store2.doc_id_map:
                 print(f"Store2 doc_id_map loaded with {len(store2.doc_id_map)} entries: {store2.doc_id_map[:5]}...") # Print first 5
            else:
                print("Store2 doc_id_map is empty after load, which is unexpected if index has entries.")


            print("\n--- Test 3: Searching in the loaded store (store2) ---")
            query_text = "Information about fixing login issues"
            print(f"Generating query embedding for: '{query_text}'")
            query_embedding_array = emb_model.generate_embeddings([query_text])

            if query_embedding_array is not None:
                query_vec = query_embedding_array[0] # Get the single embedding vector
                print(f"\nSearching for '{query_text}' (k=3)...")
                results = store2.search_similar(query_vec, k=3)
                print("Search results (doc_id, distance):")
                if results:
                    for doc_id, dist in results:
                        print(f"  ID: {doc_id}, Distance: {dist:.4f}")
                else:
                    print("  No results found.")
            else:
                print("Failed to generate query embedding.")
        else:
            print("Store2 did not load the index correctly or index was empty. Cannot perform search test.")

        print("\n--- Test 4: Adding more embeddings to store2 and saving again ---")
        if store2.index is not None : # Check if store2 is usable
            additional_texts = ["New feature: Added dark mode toggle.", "Performance optimization for data queries."]
            additional_doc_ids = ["commit_003", "tech_note_beta"]
            print("Generating embeddings for additional content...")
            additional_embeddings = emb_model.generate_embeddings(additional_texts)
            if additional_embeddings is not None:
                store2.add_embeddings(additional_embeddings, additional_doc_ids)
                store2.save_index() # Save after adding more

                print("\n--- Test 5: Creating store3 and loading the updated index ---")
                store3 = FaissVectorStore(index_file_path=test_index_file, dimension=store_dimension)
                if store3.index is not None and store3.index.ntotal > store1.index.ntotal : # type: ignore
                    print(f"Store3 loaded updated index successfully. Index has {store3.index.ntotal} entries (expected {store1.index.ntotal + len(additional_texts)}).") # type: ignore
                    print(f"Store3 doc_id_map (last 5): {store3.doc_id_map[-5:]}")
                else:
                     print(f"Store3 did not load updated index correctly. Store3 ntotal: {store3.index.ntotal if store3.index else 'None'}, Store1 ntotal: {store1.index.ntotal if store1.index else 'None'}")
            else:
                print("Failed to generate additional embeddings.")
        else:
            print("Skipping Test 4 & 5 as store2 index is not available.")


    except Exception as e:
        print(f"An error occurred in FaissVectorStore __main__ example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test files after execution
        print(f"\nCleaning up test files: {test_index_file} and {test_index_file}.map...")
        if os.path.exists(test_index_file): os.remove(test_index_file)
        if os.path.exists(test_index_file + ".map"): os.remove(test_index_file + ".map")
        print("Cleanup complete.")

    print("\n--- FaissVectorStore __main__ Test Complete ---")