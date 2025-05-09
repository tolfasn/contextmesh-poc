# core/embedding_generator.py
# This module will generate text embeddings.
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'): # MODIFIED MODEL NAME
        """
        Initializes the EmbeddingModel.
        :param model_name: Name of the sentence-transformer model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
            # Get embedding dimension dynamically
            self.dimension = self.model.get_sentence_embedding_dimension()
            if self.dimension is None: # Should not happen with standard models
                raise ValueError("Could not determine embedding dimension from model.")
            print(f"Embedding model '{model_name}' loaded. Dimension: {self.dimension}")
        except Exception as e:
            print(f"Error loading sentence transformer model '{model_name}': {e}")
            raise # Re-raise the exception to halt if model loading fails

    def generate_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """
        Generates embeddings for a list of texts.
        Returns a NumPy array of embeddings, or None if input is empty or error occurs.
        """
        if not texts or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            print("Input 'texts' must be a non-empty list of strings.")
            return None
        try:
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            print("Embeddings generated successfully.")
            return embeddings # type: ignore
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None

if __name__ == '__main__':
    try:
        print("--- EmbeddingModel __main__ Test ---")
        # Initialize with the corrected default model name
        model = EmbeddingModel() 

        sample_texts = [
            "This is a test commit message about fixing a critical bug.",
            "Refactored the authentication module for better security.",
            "Initial commit: project setup.",
            "" 
        ]

        print(f"\nAttempting to generate embeddings for {len(sample_texts)} sample texts...")
        embeddings = model.generate_embeddings(sample_texts)

        if embeddings is not None:
            print(f"\nSuccessfully generated embeddings. Shape of embeddings array: {embeddings.shape}")
            for i, text in enumerate(sample_texts):
                norm = np.linalg.norm(embeddings[i])
                print(f"  Text: '{text[:40].replace(chr(10), ' ')}...'")
                print(f"    Embedding (first 5 dims): {embeddings[i][:5]}...")
                print(f"    Embedding shape: {embeddings[i].shape}, L2 Norm: {norm:.4f}")
        else:
            print("\nFailed to generate embeddings in the example.")

        print("\n--- Testing with empty list ---")
        empty_list_embeddings = model.generate_embeddings([])
        if empty_list_embeddings is None:
            print("Correctly handled empty list input (returned None).")
        else:
            print(f"Incorrectly handled empty list, got: {empty_list_embeddings}")

        print("\n--- Testing with invalid input type ---")
        invalid_input_embeddings = model.generate_embeddings([123, "text"]) # type: ignore
        if invalid_input_embeddings is None:
            print("Correctly handled invalid input type in list (returned None).")
        else:
            print(f"Incorrectly handled invalid input, got: {invalid_input_embeddings}")

    except Exception as e:
        print(f"An error occurred in EmbeddingModel __main__ example: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- EmbeddingModel __main__ Test Complete ---")