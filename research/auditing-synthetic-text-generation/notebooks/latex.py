import os
import tempfile
import shutil
import git

from dotenv import load_dotenv
from pathlib import Path
from typing import Optional


class Project:
    def __init__(self, path: str):
        """
        Initialize the Latex class with the given path.

        Args:
            path (str): The path to the directory where the LaTeX files are located.

        Raises:
            ValueError: If the provided path does not exist.
        """
        # check if path is local
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        self.dir = Path(path)
        self.repo = git.Repo(path)

    def add_file(self, source: str, target: str):
        """
        Add a file to the repository by copying it from the source location to the target location.
        Args:
            source (str): The path to the source file that needs to be copied.
            target (str): The target path within the repository where the file should be copied.
        Returns:
            None
        """
        # Copy the source file to the target location in the repository
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        
        self.repo.index.add([str(target_path)])
        
    def add_matplotlib(self, fig, target: str, **kwargs):
        """
        Save a Matplotlib figure as a PNG file and add it to the repository index.
        Parameters:
        fig (matplotlib.figure.Figure): The Matplotlib figure to save.
        target (str): The target file path where the figure will be saved.
        **kwargs: Additional keyword arguments to pass to `fig.savefig`.
        Notes:
        - The target path will be created if it does not exist.
        - The figure will be saved with `bbox_inches` set to "tight" by default, unless overridden in `kwargs`.
        - The saved file will be added to the repository index.
        """
        kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        # Save the matplotlib figure as a PNG file
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, **kwargs)
        
        self.repo.index.add([str(target_path)])

    def add_dataframe_as_tsv(self, df, target: str, **kwargs):
        """
        Adds a pandas DataFrame to the repository by saving it as a TSV file and staging it for commit.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        target (str): The target file path where the DataFrame will be saved.
        **kwargs: Additional keyword arguments to pass to `pandas.DataFrame.to_csv`.

        Keyword Args:
        sep (str): Field delimiter for the output file. Defaults to '\t'.
        index (bool): Whether to write row names (index). Defaults to False.

        Returns:
        None
        """
        kwargs["sep"] = kwargs.get("sep", "\t")
        kwargs["index"] = kwargs.get("index", False)

        # Save the pandas DataFrame as a TSV file
        target_path = self.dir / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, **kwargs)

        self.repo.index.add([str(target_path)])
        