# core/git_interface.py
# This module will handle interactions with Git repositories.
import pygit2 # type: ignore
import os
from datetime import datetime, timezone

class GitProcessor:
    def __init__(self, repo_path: str):
        """
        Initializes the GitProcessor.
        :param repo_path: Path to the Git repository.
        """
        self.repo_path = os.path.abspath(repo_path)
        try:
            # Attempt to open the repository to check if it's valid
            self.repo = pygit2.Repository(self.repo_path)
        except pygit2.GitError:
            raise ValueError(f"Could not open or find Git repository at {self.repo_path}")

    def get_commit_data(self, commit_sha: str):
        """
        Retrieves data for a specific commit.
        Returns a dictionary with commit details or None if not found.
        """
        try:
            commit = self.repo.get(commit_sha)
            if not isinstance(commit, pygit2.Commit):
                return None

            commit_time = datetime.fromtimestamp(commit.commit_time, timezone.utc)

            return {
                "sha": commit.hex,
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "timestamp_utc": commit_time.isoformat(),
                "message": commit.message.strip(),
                # "parents": [p.hex for p in commit.parents],
                # Add diff later if needed
            }
        except (KeyError, pygit2.GitError, TypeError): # TypeError for invalid SHA format
            return None


    def get_commit_history(self, max_commits=100):
        """
        Parses the commit history of the repository.
        Returns a list of dictionaries, each representing a commit.
        :param max_commits: Maximum number of commits to retrieve.
        """
        print(f"Parsing commit history for {self.repo_path} (max {max_commits} commits)...")
        commits_data = []
        try:
            for i, commit in enumerate(self.repo.walk(self.repo.head.target, pygit2.GIT_SORT_TIME)):
                if i >= max_commits:
                    break

                commit_time = datetime.fromtimestamp(commit.commit_time, timezone.utc)

                commits_data.append({
                    "sha": commit.hex,
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "timestamp_utc": commit_time.isoformat(),
                    "message": commit.message.strip(),
                })
            print(f"Retrieved {len(commits_data)} commits.")
        except Exception as e:
            print(f"Error walking commit history: {e}")
        return commits_data

if __name__ == '__main__':
    # Example usage (for testing purposes when this file is run directly)
    # Ensure you run this from a directory that IS a git repo, or provide a valid path.
    try:
        # Test with the current directory if it's a Git repo
        # For this to work, you'd typically run `python core/git_interface.py`
        # from the root of a git repository.
        # For robust testing, create a dummy repo or use a known one.
        print("Attempting to process current directory as a Git repo for __main__ example...")
        # A safer path for testing might be your project's own .git directory if you're in contextmesh-poc
        # For example: processor = GitProcessor('.')
        # However, let's use a placeholder path to avoid accidental errors on user's system
        # and instruct user to change it for real testing.

        # IMPORTANT: For this __main__ block to work, you need to be inside a Git repository
        # or provide a valid path to one.
        # For example, if `contextmesh-poc` is a git repo, run this from `contextmesh-poc` as:
        # python core/git_interface.py

        # processor = GitProcessor('.') # Assumes current dir is a git repo
        # history = processor.get_commit_history(max_commits=5)
        # if history:
        #     print(f"\nLast {len(history)} commits:")
        #     for c_data in history:
        #         print(f"  SHA: {c_data['sha'][:7]}, Author: {c_data['author_name']}, Date: {c_data['timestamp_utc']}")
        #         print(f"  Msg: {c_data['message'][:70]}...")

        #     specific_commit_sha = history[0]['sha'] # Get SHA of the latest commit
        #     print(f"\nDetails for commit {specific_commit_sha[:7]}:")
        #     commit_details = processor.get_commit_data(specific_commit_sha)
        #     if commit_details:
        #         for key, value in commit_details.items():
        #             print(f"  {key.replace('_', ' ').title()}: {value}")
        # else:
        #     print("No commit history found or current directory is not a Git repository.")
        print("GitProcessor __main__ example: Uncomment and adapt path to test.")

    except ValueError as ve:
        print(f"Error in example: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in example: {e}")
    pass