# contextmesh_cli.py
import click
import os

# Import our core modules
from core.git_interface import GitProcessor
from core.embedding_generator import EmbeddingModel
from core.vector_store_manager import FaissVectorStore

# Default path for the FAISS index file (can be made configurable later)
DEFAULT_INDEX_FILE = "contextmesh_index.faiss"

@click.group()
def cli():
    """
    ContextMesh PoC: AI-powered developer intelligence.
    Query the 'why' behind code changes.
    """
    pass

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--max-commits', default=1000, help="Maximum number of commits to process from history.", type=int)
@click.option('--index-file', default=DEFAULT_INDEX_FILE, help="Path to the FAISS index file.", type=click.Path())
def index(repo_path: str, max_commits: int, index_file: str):
    """
    Indexes a Git repository.
    Extracts commit messages, generates embeddings, and stores them.
    REPO_PATH: Path to the Git repository to index.
    """
    click.echo(f"Starting to index repository at: {repo_path}")
    click.echo(f"Processing up to {max_commits} commits.")
    click.echo(f"Using FAISS index file: {index_file}")

    try:
        # 1. Initialize GitProcessor
        click.echo("Initializing Git processor...")
        git_processor = GitProcessor(repo_path)

        # 2. Get commit history
        click.echo("Fetching commit history...")
        # For PoC, let's get commit messages and their SHAs
        # We'll use the full commit message for embedding
        commits = git_processor.get_commit_history(max_commits=max_commits)

        if not commits:
            click.echo("No commits found or error fetching history. Nothing to index.")
            return

        click.echo(f"Retrieved {len(commits)} commits to process.")

        commit_texts_to_embed = []
        commit_shas_for_ids = []

        for commit_data in commits:
            # We'll use the full commit message for embedding
            # In a more advanced version, we might concatenate title + body, or process diffs
            full_message = commit_data.get("message_full", "").strip()
            sha = commit_data.get("sha", None)

            if full_message and sha: # Only process if we have a message and SHA
                commit_texts_to_embed.append(full_message)
                commit_shas_for_ids.append(sha)
            else:
                click.echo(f"Skipping commit due to missing message or SHA: {commit_data.get('sha', 'Unknown SHA')[:7]}")

        if not commit_texts_to_embed:
            click.echo("No valid commit messages found to embed after filtering. Indexing aborted.")
            return

        click.echo(f"Prepared {len(commit_texts_to_embed)} commit messages for embedding.")

        # 3. Initialize EmbeddingModel
        click.echo("Initializing embedding model...")
        # This will download the model on first run if not cached
        embedding_model = EmbeddingModel() # Uses default 'BAAI/bge-small-en-v1.5'

        # Get the dimension from the loaded embedding model
        embedding_dimension = embedding_model.dimension
        click.echo(f"Embedding model loaded. Dimension: {embedding_dimension}")

        # 4. Generate embeddings
        click.echo("Generating embeddings for commit messages...")
        embeddings_array = embedding_model.generate_embeddings(commit_texts_to_embed)

        if embeddings_array is None or len(embeddings_array) == 0:
            click.echo("Failed to generate embeddings. Indexing aborted.")
            return

        click.echo(f"Successfully generated {len(embeddings_array)} embeddings.")

        # 5. Initialize FaissVectorStore
        # The FaissVectorStore __init__ will try to load an existing index
        # or prepare for a new one.
        click.echo(f"Initializing FAISS vector store (dimension: {embedding_dimension})...")
        vector_store = FaissVectorStore(index_file_path=index_file, dimension=embedding_dimension)

        # 6. Add embeddings to the vector store
        click.echo("Adding embeddings to vector store...")
        vector_store.add_embeddings(embeddings_array, commit_shas_for_ids)

        # 7. Save the index
        click.echo("Saving FAISS index and document map...")
        vector_store.save_index()

        click.echo(f"Successfully indexed {len(commit_shas_for_ids)} commits into '{index_file}'.")

    except ValueError as ve:
        click.echo(f"Error during indexing: {ve}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred during indexing: {e}", err=True)
        # For debugging, you might want to print the full traceback
        # import traceback
        # click.echo(traceback.format_exc(), err=True)

@cli.command()
@click.argument('query_or_sha')
def why(query_or_sha: str):
    """
    Analyzes a Git commit SHA or a natural language query about code.
    (Placeholder - to be implemented in Module 7)
    """
    click.echo(f"ContextMesh PoC: Analyzing '{query_or_sha}'...")
    click.echo("The 'why' command is not yet fully implemented.")
    click.echo("It will eventually:")
    click.echo("1. If SHA, get commit details. If query, generate query embedding.")
    click.echo("2. Search vector store for relevant context (commit embeddings).")
    click.echo("3. Construct prompt with query and context.")
    click.echo("4. Query LLM via OllamaConnector.")
    click.echo("5. Display LLM's answer.")
    # Example:
    # if len(query_or_sha) == 40: # Basic check for SHA-1
    #     click.echo(f"Interpreting '{query_or_sha}' as a commit SHA.")
    # else:
    #     click.echo(f"Interpreting '{query_or_sha}' as a natural language query.")
    pass

if __name__ == '__main__':
    cli()