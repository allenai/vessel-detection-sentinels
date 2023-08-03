import glob
import os
import typing as t


def delete_scratch(file_names: t.List[str], scratch_dir_path: str) -> None:
    """Delete files in specified directory, and subsequently directory itself if empty.

    Parameters
    ----------
    file_names: list[str]
        List of filenames in scratch directory.

    scratch_dir_path: str
        Path to scratch directory.

    Returns
    -------
    : None
    """
    for filename in glob.glob(file_names):
        os.remove(filename)
    try:
        os.rmdir(scratch_dir_path)
    except OSError:
        pass
    return None
