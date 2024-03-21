import importlib.metadata
import pathlib
import subprocess

__version__ = importlib.metadata.version("varipeps")

try:
    git_commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        .decode("ascii")
        .strip()
    )
except subprocess.CalledProcessError:
    git_commit = None

try:
    git_tag = (
        subprocess.check_output(
            ["git", "describe", "--exact-match", "--tags", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        .decode("ascii")
        .strip()
    )
except subprocess.CalledProcessError:
    git_tag = None
