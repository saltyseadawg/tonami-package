from pathlib import Path
from typing import List


def get_all_audiofiles(folder_path: str) -> List[str]:
    """Returns a list of audio files in found a folder and subdirectories."""
    # TODO: what audio file formats do we want to accomodate?
    return [str(file) for file in Path(folder_path).rglob("*.mp3")]


def get_tone_perfect_audiofiles(
    folder_path: str = "data/tone_perfect/tone_perfect_all_mp3",
    syllable: str = None,
    spkr: int = None,
    sex: str = None,
    tone: int = None,
):
    """Return a list of audio files in the folder specifed by speaker, sex,
    and tone.
    """
    filename = "**/"
    filename += syllable if syllable else "*"
    filename += f"{str(tone)}_" if tone else "?_"
    filename += f"{sex}V" if sex else "?V"
    filename += str(spkr) if spkr else "?"
    filename += "_MP3.mp3"

    return [str(file) for file in Path(folder_path).rglob(filename)]


def save_tone_perfect_to_json(
    files,
    syllable: str = None,
    spkr: int = None,
    sex: str = None,
    tone: int = None,
):
    pass


def parse_tone_perfect_file(filename):
    pass

def test_print():
    print("hello")
