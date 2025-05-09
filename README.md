# ContextMesh PoC

AI-powered developer intelligence platform PoC.
This CLI tool helps query the "why" behind code changes by indexing Git repositories and leveraging a local LLM.

## Project Structure

- `contextmesh_cli.py`: Main CLI application using Click.
- `core/`: Directory for core logic modules.
    - `git_interface.py`: Handles Git repository interactions.
    - `embedding_generator.py`: Generates text embeddings.
    - `vector_store_manager.py`: Manages the FAISS vector store.
    - `llm_connector.py`: Connects to the local LLM (Ollama).
- `.venv/`: Python virtual environment (not committed).
- `requirements.txt`: Python dependencies.
- `.gitignore`: Specifies intentionally untracked files.
- `contextmesh_index.faiss` (and `.map`): Default FAISS index files (generated, gitignored).
- `main_prompt.txt`: Original prompt for gpt-engineer scaffolding attempt (historical).


## Setup

1.  **Prerequisites:**
    * Python 3.10+ (developed with 3.11/3.12)
    * Git
    * Ollama installed and running ([https://ollama.ai](https://ollama.ai))
    * A model pulled via Ollama (e.g., `ollama pull mistral`)

2.  **Create and Activate Virtual Environment:**
    (From the project root directory `contextmesh-poc/`)
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    (On Windows, activation is `.venv\Scripts\activate`)

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage Examples

(Ensure your virtual environment is activated and Ollama is running with the 'mistral' model or your chosen default)

All commands are run from the `contextmesh-poc` root directory, unless a different `repo_path` is specified.

1.  **Index the current repository (e.g., `contextmesh-poc` itself):**
    ```bash
    python contextmesh_cli.py index .
    ```
    This will create/overwrite `contextmesh_index.faiss` and `contextmesh_index.faiss.map` in the current directory.

2.  **Index a different local repository:**
    ```bash
    # Example: python contextmesh_cli.py index /path/to/your/other_repo --index-file other_repo_index.faiss
    python contextmesh_cli.py index /path/to/another/local/git/repository --index-file custom_index_name.faiss
    ```
    This creates `custom_index_name.faiss` and its `.map` file.

3.  **Ask 'why' about a commit SHA from the current repository's default index:**
    ```bash
    # First, find a commit SHA from the current repo using: git log --oneline
    # Example SHA (use an actual one from your repo): bf2f015
    python contextmesh_cli.py why bf2f015
    ```

4.  **Ask 'why' with a natural language query (current repo's default index):**
    ```bash
    python contextmesh_cli.py why "What were the initial setup files?"
    ```
    ```bash
    python contextmesh_cli.py why "explain changes to the git interface" --k-results 2
    ```

5.  **Ask 'why' about a different indexed repository, specifying its index and path:**
    ```bash
    python contextmesh_cli.py why "any recent changes in the other project" --repo-path /path/to/another/local/git/repository --index-file custom_index_name.faiss
    ```

## How it Works (PoC Overview)

1.  The `index` command:
    * Uses `GitProcessor` to read commit history (messages, SHAs) from the specified Git repository.
    * Uses `EmbeddingModel` (with a sentence-transformer like `BAAI/bge-small-en-v1.5`) to convert commit messages into numerical vector embeddings.
    * Stores these embeddings and their corresponding commit SHAs in a FAISS vector index using `FaissVectorStore`. The index and a map file are saved to disk.

2.  The `why` command:
    * Takes a user query (either a commit SHA or natural language).
    * If it's a natural language query, `EmbeddingModel` generates an embedding for it.
    * `FaissVectorStore` searches the loaded index for commit messages with embeddings most similar to the query embedding (or uses the commit message of a provided SHA as primary context).
    * `GitProcessor` retrieves full details for these relevant commits.
    * A prompt is constructed containing the user's original query and the text of the retrieved commit messages.
    * `OllamaConnector` sends this prompt to a local LLM (e.g., Mistral).
    * The LLM's generated answer is displayed to the user.