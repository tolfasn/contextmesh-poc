# ContextMesh PoC

AI-powered developer intelligence platform PoC.
This CLI tool helps query the "why" behind code changes.

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
- `contextmesh_poc.faiss` (and `.map`): FAISS index files (generated, gitignored).

## Setup

1.  **Prerequisites:**
    * Python 3.10+
    * Git
    * Ollama installed and running ([https://ollama.ai](https://ollama.ai))
    * A model pulled via Ollama (e.g., `ollama pull mistral`)

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

(To be run from the `contextmesh-poc` root directory with virtual environment activated)

Basic command structure:
```bash
python contextmesh_cli.py [COMMAND] [ARGS]...