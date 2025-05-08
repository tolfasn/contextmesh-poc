# core/embedding_generator.py
# This module will generate text embeddings.
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='bge-small-en-v1.5'):
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
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """
        Generates embeddings for a list of texts.
        Returns a NumPy array of embeddings, or None if input is empty or error occurs.
        """
        if not texts or not isinstance(texts, list):
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
    # Example usage
    try:
        model = EmbeddingModel() # Uses default 'bge-small-en-v1.5'
        print(f"Model dimension: {model.dimension}")
        sample_texts = [
            "This is a test commit message about fixing a critical bug.",
            "Refactored the authentication module for better security.",
            "Initial commit: project setup."
        ]
        embeddings = model.generate_embeddings(sample_texts)

        if embeddings is not None:
            for i, text in enumerate(sample_texts):
                # print first 5 dimensions as an example
                print(f"Embedding for '{text[:30]}...': {embeddings[i][:5]}... Shape: {embeddings[i].shape}")
            print(f"Shape of embeddings array: {embeddings.shape}")
        else:
            print("Failed to generate embeddings in example.")

    except Exception as e:
        print(f"Error in EmbeddingModel example: {e}")