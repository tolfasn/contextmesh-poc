# contextmesh_cli.py
import click
import os
import re # For basic SHA detection

# Import our core modules
from core.git_interface import GitProcessor
from core.embedding_generator import EmbeddingModel
from core.vector_store_manager import FaissVectorStore
from core.llm_connector import OllamaConnector 

# Configuration constants
DEFAULT_INDEX_FILE = "contextmesh_index.faiss" # Default name for the main index
DEFAULT_MODEL_NAME = "mistral" # Default Ollama model for 'why' command
DEFAULT_K_RESULTS = 3 # Default number of context results for 'why' command

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """
    ContextMesh PoC: AI-powered developer intelligence.
    Query the 'why' behind code changes by indexing Git repositories
    and leveraging a local LLM.
    """
    pass

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--max-commits', default=1000, show_default=True, help="Maximum number of commits to process from history.", type=int)
@click.option('--index-file', default=DEFAULT_INDEX_FILE, show_default=True, help="Path to the FAISS index file to create/overwrite.", type=click.Path())
def index(repo_path: str, max_commits: int, index_file: str):
    """
    Indexes a Git repository. Overwrites existing index if names conflict.
    Extracts commit messages, generates embeddings, and stores them.
    
    REPO_PATH: Path to the Git repository to index (e.g., '.' for current directory).
    """
    click.echo(click.style(f"Starting to index repository at: {repo_path}", fg='cyan'))
    click.echo(f"Processing up to {max_commits} commits.")
    click.echo(f"Using FAISS index file: {index_file}. This will overwrite any existing index with this name.")

    index_map_file = index_file + ".map"
    for f_path in [index_file, index_map_file]:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                click.echo(f"Removed existing file: {f_path}")
            except OSError as e:
                click.echo(f"Warning: Could not remove existing file {f_path}: {e}", err=True)
    
    try:
        click.echo("Initializing Git processor...")
        git_processor = GitProcessor(repo_path)

        click.echo("Fetching commit history...")
        commits = git_processor.get_commit_history(max_commits=max_commits)

        if not commits:
            click.secho("No commits found or error fetching history. Nothing to index.", fg='yellow')
            return

        click.echo(f"Retrieved {len(commits)} commits to process.")
        commit_texts_to_embed = []
        commit_shas_for_ids = []

        for commit_data in commits:
            full_message = commit_data.get("message_full", "").strip()
            sha = commit_data.get("sha", None)
            if full_message and sha: # Only process if we have a non-empty message and a SHA
                commit_texts_to_embed.append(full_message)
                commit_shas_for_ids.append(sha)
            else:
                click.echo(f"Skipping commit due to missing message or SHA: {commit_data.get('sha', 'Unknown SHA')[:7]}", fg='yellow')
        
        if not commit_texts_to_embed:
            click.secho("No valid commit messages found to embed after filtering. Indexing aborted.", fg='yellow')
            return

        click.echo(f"Prepared {len(commit_texts_to_embed)} commit messages for embedding.")
        click.echo("Initializing embedding model (this may download model files on first run)...")
        embedding_model = EmbeddingModel()
        embedding_dimension = embedding_model.dimension
        click.echo(f"Embedding model loaded. Dimension: {embedding_dimension}")

        click.echo("Generating embeddings for commit messages...")
        embeddings_array = embedding_model.generate_embeddings(commit_texts_to_embed)

        if embeddings_array is None or len(embeddings_array) == 0:
            click.secho("Failed to generate embeddings. Indexing aborted.", fg='red')
            return
        
        click.echo(f"Successfully generated {len(embeddings_array)} embeddings.")
        
        click.echo(f"Initializing FAISS vector store (dimension: {embedding_dimension})...")
        vector_store = FaissVectorStore(index_file_path=index_file, dimension=embedding_dimension)
        
        click.echo("Adding embeddings to vector store...")
        vector_store.add_embeddings(embeddings_array, commit_shas_for_ids)
        
        click.echo("Saving FAISS index and document map...")
        vector_store.save_index()
        click.secho(f"Successfully indexed {len(commit_shas_for_ids)} commits into '{index_file}'.", fg='green')

    except ValueError as ve: # Catch ValueErrors from our modules (e.g., bad repo path)
        click.secho(f"Error during indexing: {ve}", fg='red', err=True)
    except Exception as e: # Catch any other unexpected errors
        click.secho(f"An unexpected error occurred during indexing: {e}", fg='red', err=True)
        # For deeper debugging:
        # import traceback
        # click.echo(traceback.format_exc(), err=True)

@cli.command()
@click.argument('query_or_sha')
@click.option('--repo-path', default='.', show_default=True, help="Path to the Git repository to query against.", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--index-file', default=DEFAULT_INDEX_FILE, show_default=True, help="Path to the FAISS index file to use.", type=click.Path(exists=True, file_okay=True, dir_okay=False)) # Index file must exist for 'why'
@click.option('--k-results', default=DEFAULT_K_RESULTS, show_default=True, help="Number of relevant commit messages to retrieve for context.", type=int)
@click.option('--llm-model', default=DEFAULT_MODEL_NAME, show_default=True, help="Name of the Ollama model to use.")
def why(query_or_sha: str, repo_path: str, index_file: str, k_results: int, llm_model: str):
    """
    Explains the 'why' behind a commit SHA or a code-related query
    using the indexed Git data and a local LLM.
    
    QUERY_OR_SHA: A Git commit SHA (short or full) or a natural language question.
    """
    click.echo(click.style(f"Received query/SHA: '{query_or_sha}'", fg='cyan'))
    click.echo(f"Using repository: {repo_path}")
    click.echo(f"Using index file: {index_file}")

    try:
        click.echo("Initializing components...")
        embedding_model = EmbeddingModel() 
        embedding_dimension = embedding_model.dimension
        
        vector_store = FaissVectorStore(index_file_path=index_file, dimension=embedding_dimension)
        # load_index() is called in FaissVectorStore's __init__
        if vector_store.index is None or vector_store.index.ntotal == 0:
            click.secho(f"Error: Index file '{index_file}' is empty or could not be loaded properly. Please run the 'index' command first for this repository and index file.", fg='red', err=True)
            return

        git_processor = GitProcessor(repo_path) 
        llm_connector = OllamaConnector() 

        is_sha = bool(re.fullmatch(r'[0-9a-f]{4,40}', query_or_sha, re.IGNORECASE)) # Relaxed SHA check (4-40 chars)
        
        query_text_for_llm = query_or_sha 
        context_commit_messages = []

        if is_sha:
            click.echo(f"Interpreting '{query_or_sha}' as a commit SHA.")
            commit_data = git_processor.get_commit_data(query_or_sha)
            if not commit_data:
                click.secho(f"Error: Could not find commit data for SHA '{query_or_sha}'. Ensure it's a valid SHA in the repository '{repo_path}'.", fg='red', err=True)
                return
            
            click.echo(f"Found commit: {commit_data['sha'][:7]} - {commit_data['message_short']}")
            # Primary context is the commit itself
            context_commit_messages.append(f"Primary Commit SHA: {commit_data['sha']}\nAuthor: {commit_data['author_name']}\nDate: {commit_data['author_timestamp_utc']}\nMessage:\n{commit_data['message_full']}\n---")
            
            # Find semantically similar commits to this one's message for broader context
            if commit_data['message_full']: # Only if there's a message to embed
                query_embedding = embedding_model.generate_embeddings([commit_data['message_full']])
                if query_embedding is not None and k_results > 1: # Only search if k_results allows for more
                    click.echo(f"Finding up to {k_results-1} additional commits semantically similar to SHA {query_or_sha[:7]}...")
                    similar_commits = vector_store.search_similar(query_embedding[0], k=k_results) # Fetch k, then filter
                    for sha, dist in similar_commits:
                        if sha != commit_data['sha'] and len(context_commit_messages) < k_results: 
                            c_data = git_processor.get_commit_data(sha)
                            if c_data:
                                context_commit_messages.append(f"Related Commit SHA: {c_data['sha']}\nAuthor: {c_data['author_name']}\nDate: {c_data['author_timestamp_utc']}\nMessage:\n{c_data['message_full']}\n---")
            query_text_for_llm = f"Explain the rationale, purpose, or context behind commit {commit_data['sha'][:7]} ('{commit_data['message_short']}'). Focus on why this change might have been made."

        else: # Natural language query
            click.echo(f"Interpreting '{query_or_sha}' as a natural language query.")
            click.echo("Generating embedding for the query...")
            query_embedding = embedding_model.generate_embeddings([query_or_sha])

            if query_embedding is None:
                click.secho("Error: Could not generate embedding for the query.", fg='red', err=True)
                return

            click.echo(f"Searching for {k_results} relevant commit messages in the index...")
            relevant_shas_with_scores = vector_store.search_similar(query_embedding[0], k=k_results)
            
            if not relevant_shas_with_scores:
                click.secho("No relevant commits found in the index for your query.", fg='yellow')
                # Consider if we should still query LLM with just the question. For PoC, context is key.
                return

            click.echo(f"Found {len(relevant_shas_with_scores)} relevant commit(s): {', '.join([s[:7] for s, _ in relevant_shas_with_scores])}")
            click.echo("Fetching details for relevant commits...")
            for sha, score in relevant_shas_with_scores:
                commit_data = git_processor.get_commit_data(sha)
                if commit_data:
                    context_commit_messages.append(f"Commit SHA: {commit_data['sha']} (Similarity Score: {1-score:.4f})\nAuthor: {commit_data['author_name']}\nDate: {commit_data['author_timestamp_utc']}\nMessage:\n{commit_data['message_full']}\n---")
                else:
                    click.echo(f"Warning: Could not retrieve details for commit SHA {sha}", fg='yellow', err=True)
        
        if not context_commit_messages:
            click.secho("Could not gather any context from commits. Aborting LLM query.", fg='red', err=True)
            return

        click.echo("Constructing prompt for LLM...")
        context_str = "\n\n".join(context_commit_messages)
        
        llm_prompt = (
            f"You are ContextMesh, an AI assistant helping a software developer understand their codebase.\n"
            f"Your task is to answer the developer's question or explain a commit based ONLY on the provided context from Git commit messages below.\n"
            f"If the context is insufficient to fully answer, state that the provided commits do not contain enough detail, but still try to infer what you can.\n"
            f"Be concise and directly address the question.\n\n"
            f"=== Context from relevant Git Commits ===\n{context_str}\n\n"
            f"=== Developer's Question/Request ===\n{query_text_for_llm}\n\n"
            f"Answer:"
        )
        
        # For debugging the prompt:
        # click.echo(click.style("\n--- Generated LLM Prompt ---", bold=True))
        # click.echo(llm_prompt)
        # click.echo(click.style("---------------------------\n", bold=True))

        click.echo(f"Sending prompt to LLM model '{llm_model}' (this may take a moment)...")
        llm_response = llm_connector.query_llm(llm_model, llm_prompt, stream=True) 

        if llm_response:
            click.echo(click.style("\n=== ContextMesh AI Answer ===\n", fg='green', bold=True))
            click.echo(llm_response)
        else:
            click.secho("Error: No response received from LLM, or LLM query failed.", fg='red', err=True)

    except ValueError as ve:
        click.secho(f"Error during 'why' command: {ve}", fg='red', err=True)
    except Exception as e:
        click.secho(f"An unexpected error occurred during 'why' command: {e}", fg='red', err=True)
        # For deeper debugging:
        # import traceback
        # click.echo(traceback.format_exc(), err=True)

if __name__ == '__main__':
    cli()
