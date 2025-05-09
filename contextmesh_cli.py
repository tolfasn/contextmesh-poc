# contextmesh_cli.py
import click
import os
import re # For basic SHA detection

# Import our core modules
from core.git_interface import GitProcessor
from core.embedding_generator import EmbeddingModel
from core.vector_store_manager import FaissVectorStore
from core.llm_connector import OllamaConnector # Import the LLM connector

# Default path for the FAISS index file
DEFAULT_INDEX_FILE = "contextmesh_index.faiss"
DEFAULT_MODEL_NAME = "mistral" # Default Ollama model

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
    Indexes a Git repository. Overwrites existing index.
    Extracts commit messages, generates embeddings, and stores them.
    REPO_PATH: Path to the Git repository to index.
    """
    click.echo(f"Starting to index repository at: {repo_path}")
    click.echo(f"Processing up to {max_commits} commits.")
    click.echo(f"Using FAISS index file: {index_file}. This will overwrite any existing index with this name.")

    # --- MODIFICATION START: Delete existing index files before processing ---
    index_map_file = index_file + ".map"
    if os.path.exists(index_file):
        try:
            os.remove(index_file)
            click.echo(f"Removed existing index file: {index_file}")
        except OSError as e:
            click.echo(f"Warning: Could not remove existing index file {index_file}: {e}", err=True)
    
    if os.path.exists(index_map_file):
        try:
            os.remove(index_map_file)
            click.echo(f"Removed existing index map file: {index_map_file}")
        except OSError as e:
            click.echo(f"Warning: Could not remove existing index map file {index_map_file}: {e}", err=True)
    # --- MODIFICATION END ---

    try:
        click.echo("Initializing Git processor...")
        git_processor = GitProcessor(repo_path)

        click.echo("Fetching commit history...")
        commits = git_processor.get_commit_history(max_commits=max_commits)

        if not commits:
            click.echo("No commits found or error fetching history. Nothing to index.")
            return

        click.echo(f"Retrieved {len(commits)} commits to process.")
        commit_texts_to_embed = []
        commit_shas_for_ids = []

        for commit_data in commits:
            full_message = commit_data.get("message_full", "").strip()
            sha = commit_data.get("sha", None)
            if full_message and sha:
                commit_texts_to_embed.append(full_message)
                commit_shas_for_ids.append(sha)
            else:
                click.echo(f"Skipping commit due to missing message or SHA: {commit_data.get('sha', 'Unknown SHA')[:7]}")
        
        if not commit_texts_to_embed:
            click.echo("No valid commit messages found to embed after filtering. Indexing aborted.")
            return

        click.echo(f"Prepared {len(commit_texts_to_embed)} commit messages for embedding.")
        click.echo("Initializing embedding model...")
        embedding_model = EmbeddingModel()
        embedding_dimension = embedding_model.dimension
        click.echo(f"Embedding model loaded. Dimension: {embedding_dimension}")

        click.echo("Generating embeddings for commit messages...")
        embeddings_array = embedding_model.generate_embeddings(commit_texts_to_embed)

        if embeddings_array is None or len(embeddings_array) == 0:
            click.echo("Failed to generate embeddings. Indexing aborted.")
            return
        
        click.echo(f"Successfully generated {len(embeddings_array)} embeddings.")
        
        # Now, FaissVectorStore will always create a new index because we deleted the old files
        click.echo(f"Initializing FAISS vector store (dimension: {embedding_dimension})...")
        vector_store = FaissVectorStore(index_file_path=index_file, dimension=embedding_dimension)
        
        click.echo("Adding embeddings to vector store...")
        vector_store.add_embeddings(embeddings_array, commit_shas_for_ids)
        
        click.echo("Saving FAISS index and document map...")
        vector_store.save_index()
        click.echo(f"Successfully indexed {len(commit_shas_for_ids)} commits into '{index_file}'.")

    except ValueError as ve:
        click.echo(f"Error during indexing: {ve}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred during indexing: {e}", err=True)

@cli.command()
@click.argument('query_or_sha')
@click.option('--repo-path', default='.', help="Path to the Git repository to query against.", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--index-file', default=DEFAULT_INDEX_FILE, help="Path to the FAISS index file.", type=click.Path())
@click.option('--k-results', default=3, help="Number of relevant commit messages to retrieve for context.", type=int)
@click.option('--llm-model', default=DEFAULT_MODEL_NAME, help="Name of the Ollama model to use.")
def why(query_or_sha: str, repo_path: str, index_file: str, k_results: int, llm_model: str):
    """
    Explains the 'why' behind a commit SHA or a code-related query.
    QUERY_OR_SHA: A Git commit SHA or a natural language question.
    """
    click.echo(f"Received query/SHA: '{query_or_sha}'")
    click.echo(f"Using repository: {repo_path}")
    click.echo(f"Using index file: {index_file}")

    try:
        # 1. Initialize necessary components
        click.echo("Initializing components...")
        embedding_model = EmbeddingModel() # For query embedding
        embedding_dimension = embedding_model.dimension

        vector_store = FaissVectorStore(index_file_path=index_file, dimension=embedding_dimension)
        if vector_store.index is None or vector_store.index.ntotal == 0:
            click.echo(f"Error: Index file '{index_file}' is empty or not found. Please run the 'index' command first.", err=True)
            return

        git_processor = GitProcessor(repo_path) # For fetching commit details
        llm_connector = OllamaConnector() # For querying the LLM

        # 2. Determine if input is SHA or natural language query (basic check)
        # A simple regex for typical Git SHA (40 hex chars, or shorter common forms)
        is_sha = bool(re.fullmatch(r'[0-9a-f]{7,40}', query_or_sha, re.IGNORECASE))

        query_text_for_llm = query_or_sha # This will be part of the LLM prompt

        relevant_commit_shas = []
        context_commit_messages = []

        if is_sha:
            click.echo(f"Interpreting '{query_or_sha}' as a commit SHA.")
            # For a SHA, the primary context is the commit itself.
            # We might also want to find *related* commits semantically.
            # For PoC, let's make the SHA itself the primary context to explain.
            # We can also embed its message and find similar ones.

            commit_data = git_processor.get_commit_data(query_or_sha)
            if not commit_data:
                click.echo(f"Error: Could not find commit data for SHA '{query_or_sha}'.", err=True)
                return

            click.echo(f"Found commit: {commit_data['sha'][:7]} - {commit_data['message_short']}")
            context_commit_messages.append(f"Commit SHA: {commit_data['sha']}\nAuthor: {commit_data['author_name']}\nDate: {commit_data['author_timestamp_utc']}\nMessage:\n{commit_data['message_full']}\n---")

            # Optionally, also find semantically similar commits to this one's message
            query_embedding = embedding_model.generate_embeddings([commit_data['message_full']])
            if query_embedding is not None:
                click.echo(f"Finding {k_results-1} additional commits semantically similar to SHA {query_or_sha[:7]}...")
                # k_results-1 because we already have the primary commit
                similar_commits = vector_store.search_similar(query_embedding[0], k=max(1, k_results)) # ensure k is at least 1
                for sha, dist in similar_commits:
                    if sha != query_or_sha and len(context_commit_messages) < k_results : # Avoid duplicate and limit results
                        c_data = git_processor.get_commit_data(sha)
                        if c_data:
                            context_commit_messages.append(f"Related Commit SHA: {c_data['sha']}\nAuthor: {c_data['author_name']}\nDate: {c_data['author_timestamp_utc']}\nMessage:\n{c_data['message_full']}\n---")
            query_text_for_llm = f"Explain the rationale or context behind commit {query_or_sha[:7]}: '{commit_data['message_short']}'"


        else: # Natural language query
            click.echo(f"Interpreting '{query_or_sha}' as a natural language query.")
            click.echo("Generating embedding for the query...")
            query_embedding = embedding_model.generate_embeddings([query_or_sha])

            if query_embedding is None:
                click.echo("Error: Could not generate embedding for the query.", err=True)
                return

            click.echo(f"Searching for {k_results} relevant commit messages in the index...")
            # Search results are [(doc_id, distance_score), ...]
            # doc_id in our case is the commit SHA
            relevant_commit_shas_with_scores = vector_store.search_similar(query_embedding[0], k=k_results)

            if not relevant_commit_shas_with_scores:
                click.echo("No relevant commits found in the index for your query.")
                # We could still try to query the LLM with just the user's question,
                # but for this PoC, context is key.
                return

            relevant_commit_shas = [sha for sha, score in relevant_commit_shas_with_scores]
            click.echo(f"Found {len(relevant_commit_shas)} relevant commit SHAs: {', '.join([s[:7] for s in relevant_commit_shas])}")

            # 3. Retrieve content for these relevant commits
            click.echo("Fetching details for relevant commits...")
            for sha in relevant_commit_shas:
                commit_data = git_processor.get_commit_data(sha)
                if commit_data:
                    context_commit_messages.append(f"Commit SHA: {commit_data['sha']}\nAuthor: {commit_data['author_name']}\nDate: {commit_data['author_timestamp_utc']}\nMessage:\n{commit_data['message_full']}\n---")
                else:
                    click.echo(f"Warning: Could not retrieve details for commit SHA {sha}", err=True)

        if not context_commit_messages:
            click.echo("Could not gather any context from commits. Aborting LLM query.", err=True)
            return

        # 4. Construct the prompt for the LLM
        click.echo("Constructing prompt for LLM...")
        context_str = "\n\n".join(context_commit_messages)

        # Basic Prompt Engineering:
        # Provide role, task, context, and then the question.
        llm_prompt = (
            f"You are an AI assistant helping a software developer understand their codebase.\n"
            f"Your task is to answer the developer's question or explain a commit based on the provided context from Git commit messages.\n"
            f"Be concise and focus on the information present in the context.\n\n"
            f"=== Context from relevant Git Commits ===\n{context_str}\n\n"
            f"=== Developer's Question/Request ===\n{query_text_for_llm}\n\n"
            f"Answer:"
        )

        # click.echo(f"\n--- Generated LLM Prompt ---\n{llm_prompt}\n---------------------------\n") # For debugging

        # 5. Query the LLM
        click.echo(f"Sending prompt to LLM model '{llm_model}' (this may take a moment)...")
        llm_response = llm_connector.query_llm(llm_model, llm_prompt, stream=True) # Use streaming

        # 6. Display the LLM's answer
        if llm_response:
            click.echo(click.style("\n=== ContextMesh AI Answer ===\n", fg='green', bold=True))
            click.echo(llm_response)
        else:
            click.echo("Error: No response received from LLM.", err=True)

    except ValueError as ve:
        click.echo(f"Error during 'why' command: {ve}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred during 'why' command: {e}", err=True)
        # import traceback
        # click.echo(traceback.format_exc(), err=True)

if __name__ == '__main__':
    cli()