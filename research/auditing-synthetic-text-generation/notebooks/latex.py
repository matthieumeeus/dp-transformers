import os
import tempfile
import shutil
import git

from dotenv import load_dotenv
from pathlib import Path
from typing import Optional


class Project:
    def __init__(self, url: str):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp_dir.name)
        self.repo = git.Repo.clone_from(url, self.dir)

    @classmethod
    def from_env(cls, url_env_name: str, pat_env_name: Optional[str] = None):
        load_dotenv()
        url = os.getenv(url_env_name)
        pat = None
        if pat_env_name:
            pat = os.getenv(pat_env_name)

        # parse the url and extract protocol and base url
        protocol, base_url = url.split("://")
        username = None
        if "@" in base_url:
            username, base_url = base_url.split("@")

        if username and pat:
            username = f"{username}:{pat}"
        elif pat:
            username = f"{pat}"

        if username:
            url = f"{protocol}://{username}@{base_url}"
        else:
            url = f"{protocol}://{base_url}"
        return cls(url=url)

    def __del__(self):
        self.tmp_dir.cleanup()
    
    def push(self, source: str, target: str, commit_message: str = "Pushing file from notebook"):
        """
        Pushes a file from a source location to a target location in the repository, commits the change, and pushes it to the remote repository.
        Args:
            source (str): The path to the source file that needs to be pushed.
            target (str): The target path within the repository where the file should be copied.
            commit_message (str, optional): The commit message for the change. Defaults to "Pushing file from notebook".
        Raises:
            Exception: If there is an error during the file copy, commit, or push process.
        """
        # Copy the source file to the target location in the repository
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        
        # Add the file to the repository
        self.repo.index.add([str(target_path)])
        self.repo.index.commit(commit_message)
        
        # Push the changes to the remote repository
        origin = self.repo.remote(name='origin')
        origin.push()

    def push_matplotlib(self, fig, target: str, commit_message: str = "Pushing matplotlib figure from notebook", **kwargs):
        """
        Pushes a matplotlib figure to the repository as a PNG file, commits the change, and pushes it to the remote repository.
        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object to be saved and pushed.
            target (str): The target path within the repository where the PNG file should be saved.
            commit_message (str, optional): The commit message for the change. Defaults to "Pushing matplotlib figure from notebook".
            **kwargs: Additional keyword arguments to pass to the savefig method.
        Raises:
            Exception: If there is an error during the file save, commit, or push process.
        """
        kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        # Save the matplotlib figure as a PNG file
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, **kwargs)
        
        # Add the file to the repository
        self.repo.index.add([str(target_path)])
        self.repo.index.commit(commit_message)

        # Push the changes to the remote repository
        origin = self.repo.remote(name='origin')
        origin.push()

    def push_dataframe(self, df, target: str, commit_message: str = "Pushing dataframe from notebook", **kwargs):
        """
        Pushes a pandas DataFrame to the repository as a TSV file, commits the change, and pushes it to the remote repository.

        PGFPlots uses a whitespace-separated format for data tables by default which is compatible with the TSV format.
        Args:
            df (pandas.DataFrame): The pandas DataFrame to be saved and pushed.
            target (str): The target path within the repository where the CSV file should be saved.
            commit_message (str, optional): The commit message for the change. Defaults to "Pushing dataframe from notebook".
            **kwargs: Additional keyword arguments to pass to the to_csv method.
        Raises:
            Exception: If there is an error during the file save, commit, or push process.
        """
        kwargs["sep"] = kwargs.get("sep", "\t")
        kwargs["index"] = kwargs.get("index", False)

        # Save the pandas DataFrame as a TSV file
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, **kwargs)
        
        # Add the file to the repository
        self.repo.index.add([str(target_path)])
        self.repo.index.commit(commit_message)

        # Push the changes to the remote repository
        origin = self.repo.remote(name='origin')
        origin.push()
