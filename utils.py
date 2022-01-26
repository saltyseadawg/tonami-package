from pathlib import Path
from typing import List


def get_audiofiles(folder_path: str) -> List[str]:
    """Returns a list of audio files in found a folder and subdirectories."""
    # TODO: what audio file formats do we want to accomodate?
    return [str(file) for file in Path(folder_path).rglob('*.mp3')]

