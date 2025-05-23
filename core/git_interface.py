# core/git_interface.py
# This module will handle interactions with Git repositories.
import pygit2 # type: ignore
import os
from datetime import datetime, timezone, timedelta

class GitProcessor:
    def __init__(self, repo_path: str):
        """
        Initializes the GitProcessor.
        :param repo_path: Path to the Git repository.
                          Can be an absolute path or relative to the current working directory.
        """
        self.repo_path = os.path.abspath(os.path.expanduser(repo_path))
        
        try:
            if not os.path.exists(self.repo_path):
                raise ValueError(f"Repository path does not exist: {self.repo_path}")
            if not os.path.isdir(self.repo_path):
                raise ValueError(f"Repository path is not a directory: {self.repo_path}")

            git_dir = pygit2.discover_repository(self.repo_path)
            if git_dir is None:
                raise ValueError(f"Could not find .git directory in or above: {self.repo_path}")
            
            self.repo = pygit2.Repository(git_dir)
            print(f"Successfully opened Git repository at: {self.repo.workdir}")

        except pygit2.GitError as e:
            raise ValueError(f"Error opening Git repository at '{self.repo_path}': {e}")
        except ValueError as ve: # Re-raise ValueErrors we explicitly raised
            raise ve
        except Exception as e: # Catch any other unexpected errors during init
            raise ValueError(f"An unexpected error occurred initializing GitProcessor for '{self.repo_path}': {e}")

    def get_commit_data(self, commit_sha_or_ref: str) -> dict | None:
        """
        Retrieves data for a specific commit SHA or a reference like 'HEAD'.
        Returns a dictionary with commit details or None if not found/invalid.
        """
        if not self.repo:
            print("Error: GitProcessor's repository is not initialized.")
            return None
        try:
            # Resolve reference (like 'HEAD', 'master', or a tag) to a commit object
            commit_obj = self.repo.revparse_single(commit_sha_or_ref)
            # Ensure we have a commit object, peeling tags if necessary
            if not isinstance(commit_obj, pygit2.Commit):
                target_commit = commit_obj.peel(pygit2.Commit) if hasattr(commit_obj, 'peel') else None
                if not isinstance(target_commit, pygit2.Commit):
                     print(f"Warning: '{commit_sha_or_ref}' did not resolve to a direct commit.")
                     return None
                commit_obj = target_commit

        except (KeyError, pygit2.GitError, TypeError) as e: # TypeError for invalid SHA format to revparse_single
            print(f"Error resolving commit SHA/ref '{commit_sha_or_ref}': {e}")
            return None
        
        if commit_obj is None: # Should be caught by above, but as a safeguard
            return None

        # Get author and committer timezone-aware datetimes
        author_time = datetime.fromtimestamp(commit_obj.author.time, 
                                             timezone(timedelta(minutes=commit_obj.author.offset)))
        committer_time = datetime.fromtimestamp(commit_obj.committer.time, 
                                                timezone(timedelta(minutes=commit_obj.committer.offset)))
        
        parents_shas = [str(p.id) for p in commit_obj.parents] 

        diff_text_parts = []
        try:
            if commit_obj.parents:
                parent_commit = commit_obj.parents[0]
                # Diff trees for changes in this commit compared to its first parent
                diff_obj = self.repo.diff(parent_commit.tree, commit_obj.tree, context_lines=0, interhunk_lines=0)
            else:
                # Initial commit, diff against an empty tree
                empty_tree_oid = self.repo.TreeBuilder().write() # Create an OID for an empty tree
                empty_tree = self.repo[empty_tree_oid] # Get the empty tree object
                diff_obj = self.repo.diff(empty_tree, commit_obj.tree, context_lines=0, interhunk_lines=0)

            if diff_obj:
                for patch_obj in diff_obj: 
                    delta = patch_obj.delta 

                    file_path_str = "[Unknown file path]" # Default
                    if delta.new_file and delta.new_file.path:
                        file_path_str = delta.new_file.path
                    elif delta.old_file and delta.old_file.path: # Fallback for deleted files
                        file_path_str = delta.old_file.path
                        
                    diff_text_parts.append(f"File: {file_path_str}")
                    diff_text_parts.append(f"  Status: {delta.status_char()}")
                    
                    # Optional: Add line stats for a richer diff summary
                    if patch_obj.line_stats: # Ensure line_stats are available
                        # line_stats is a tuple: (total_context, additions, deletions)
                        additions = patch_obj.line_stats[1] 
                        deletions = patch_obj.line_stats[2]
                        if additions > 0 or deletions > 0:
                            diff_text_parts.append(f"    Lines: +{additions}/-{deletions}")
            
        except Exception as e:
            diff_text_parts.append(f"Error generating diff: {e}")

        return {
            "sha": str(commit_obj.id),
            "author_name": commit_obj.author.name,
            "author_email": commit_obj.author.email,
            "author_timestamp_utc": author_time.astimezone(timezone.utc).isoformat(), # Standardize to UTC
            "committer_name": commit_obj.committer.name,
            "committer_email": commit_obj.committer.email,
            "committer_timestamp_utc": committer_time.astimezone(timezone.utc).isoformat(), # Standardize to UTC
            "message_short": commit_obj.message.strip().split('\n', 1)[0], # First line of commit message
            "message_full": commit_obj.message.strip(),
            "parents": parents_shas,
            "tree_sha": str(commit_obj.tree.id),
            "diff_summary": "\n".join(diff_text_parts) if diff_text_parts else "No changes detected or diff not generated."
        }

    def get_commit_history(self, max_commits: int = 100, branch_name: str | None = None) -> list[dict]:
        """
        Parses the commit history of the repository from a given branch or HEAD.
        Returns a list of dictionaries, each representing a commit (using get_commit_data).
        :param max_commits: Maximum number of commits to retrieve.
        :param branch_name: Optional specific branch name (e.g., 'main', 'develop'). If None, uses HEAD.
        """
        if not self.repo:
            print("Error: GitProcessor's repository is not initialized.")
            return []
        
        commits_data = []
        try:
            start_commit_obj = None
            if branch_name:
                # Try to get branch from local, then remote if not found locally
                target_ref_local = self.repo.branches.local.get(branch_name)
                target_ref_remote = self.repo.branches.remote.get(branch_name) # Assumes remote name is 'origin' implicitly
                
                if target_ref_local:
                    start_commit_obj = target_ref_local.peel(pygit2.Commit)
                elif target_ref_remote:
                     print(f"Branch '{branch_name}' not found locally, using remote tracking branch.")
                     start_commit_obj = target_ref_remote.peel(pygit2.Commit)
                else:
                    print(f"Branch '{branch_name}' not found locally or remotely.")
                    return []
            else: # Default to HEAD
                if self.repo.is_empty or self.repo.head_is_unborn:
                    print("Repository is empty or HEAD is unborn (no commits yet).")
                    return []
                start_commit_obj = self.repo.head.peel(pygit2.Commit) # Peel HEAD to get the commit

            if not start_commit_obj: # Should be caught above, but as a safeguard
                print("Could not determine starting commit for history traversal.")
                return []

            print(f"Parsing commit history from ref '{str(start_commit_obj.id)[:7]}' (max {max_commits} commits)...") 
            
            for i, commit_walker_obj in enumerate(self.repo.walk(start_commit_obj.id, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_TOPOLOGICAL)):
                if i >= max_commits:
                    break
                
                commit_sha_str = str(commit_walker_obj.id) 
                commit_details = self.get_commit_data(commit_sha_str) # Use our detailed method
                if commit_details:
                    commits_data.append(commit_details)
                else:
                    print(f"Warning: Could not retrieve details for commit {commit_sha_str}")
                            
            print(f"Retrieved data for {len(commits_data)} commits.")
        except pygit2.GitError as e:
            print(f"GitError while walking commit history: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while walking commit history: {e}")
            # import traceback # Uncomment for more detailed debugging if needed
            # traceback.print_exc()
        return commits_data

if __name__ == '__main__':
    print("--- GitProcessor __main__ Test ---")
    # Modify this path to test with a different repository if needed
    repo_to_test = '.' # Tests the current directory
    
    print(f"Attempting to process Git repository at: {os.path.abspath(repo_to_test)}")

    try:
        processor = GitProcessor(repo_to_test)
        
        print("\n--- Testing get_commit_history (last 3 commits from HEAD) ---")
        # Test with a smaller number of commits for __main__ output brevity
        history = processor.get_commit_history(max_commits=3) 
        
        if history:
            print(f"\nFound {len(history)} commits:")
            for i, c_data in enumerate(history):
                print(f"  Commit {i+1}:")
                print(f"    SHA: {c_data['sha'][:7]}")
                print(f"    Author: {c_data['author_name']}")
                print(f"    Date: {c_data['author_timestamp_utc']}")
                print(f"    Message: {c_data['message_short']}")
                # Print diff summary for each commit in history for quick check
                print(f"    Diff Summary:")
                for line in c_data.get('diff_summary', 'Not available').splitlines():
                    print(f"      {line}")
                print("-" * 20)

            # Get full details for the very latest commit in the history list
            if history:
                print("\n    --- Full details for the latest commit in history list ---")
                # The 'sha' key from c_data is already the full SHA string
                full_latest_commit = processor.get_commit_data(history[0]['sha']) 
                if full_latest_commit:
                    for key, value in full_latest_commit.items():
                        if key == "diff_summary": 
                            print(f"    {key.replace('_', ' ').title()}:")
                            for line in value.splitlines():
                                print(f"      {line}")
                        else:
                            print(f"    {key.replace('_', ' ').title()}: {value}")
                else:
                    print("    Could not fetch full details for this commit.")
        else:
            print("No commit history found. Ensure the path is a valid Git repo with commits.")

        print("\n--- Testing get_commit_data with 'HEAD' ---")
        head_commit_data = processor.get_commit_data('HEAD')
        if head_commit_data:
            print("Details for HEAD commit:")
            for key, value in head_commit_data.items():
                if key == "diff_summary":
                     print(f"  {key.replace('_', ' ').title()}:")
                     for line in value.splitlines():
                        print(f"    {line}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print("Could not retrieve data for HEAD.")

        print("\n--- Testing get_commit_data with a potentially invalid SHA ---")
        invalid_sha_data = processor.get_commit_data('invalid_sha_here_123_nonexistent') # Made it more unique
        if not invalid_sha_data:
            print("Correctly handled invalid SHA (no data returned or error printed by method).")
        else:
            print(f"Unexpectedly got data for invalid SHA: {invalid_sha_data}")

    except ValueError as ve:
        print(f"\nValueError during GitProcessor test: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during GitProcessor test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- GitProcessor __main__ Test Complete ---")
