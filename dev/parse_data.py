from pathlib import Path

import pandas as pd
import librosa

from .load_audio import load_audio_file


def parse_toneperfect_pitch(file_path):
    """Returns a dict containing metadata and pitch for a Tone Perfect file.
    TODO: pitch retrieved with librosa, want to do same thing with parselmouth

    Args:
        file_path (str): a Tone perfect file
    Returns:
        dict: metadata and pitch information
    """
    file_dict = {}
    file_name = str(file_path)[39:]
    file_dict["filename"] = file_name
    sections = file_name.split("_")
    file_dict["sex"] = sections[1][0]
    file_dict["speaker"] = sections[1]
    file_dict["syllable"] = sections[0][: len(sections[0]) - 1]
    file_dict["tone"] = sections[0][-1]
    file_dict["database"] = "toneperfect"
    time_series, file_dict["sample"] = load_audio_file(file_path)
    file_dict["pitch_contour"], voiced_flag, voiced_probs = librosa.pyin(
        time_series, fmin=50, fmax=500
    )
    return file_dict


def write_toneperfect_pitch_data(
    folder_path="data/tone_perfect/tone_perfect_all_mp3",
    output="data/parsed/toneperfect_pitch.json",
):
    """Creates a json file containing metadata and pitch information from
    Tone Perfect files found in folder_path.

    Args:
        folder_path (str): target folder that is searched
        output (str): name of file that is created
    """

    files, db, syl, tone, sex, spkr, sample, pitch = ([] for i in range(8))
    counter = 0
    for f in Path(folder_path).rglob("*.mp3"):
        mtd = parse_toneperfect_pitch(f)
        files.append(mtd["filename"])
        db.append(mtd["database"])
        syl.append(mtd["syllable"])
        tone.append(mtd["tone"])
        sex.append(mtd["sex"])
        spkr.append(mtd["speaker"])
        sample.append(mtd["sample"])
        pitch.append(mtd["pitch_contour"])
        if counter  == 0:
            # 9840 Tone Perfect files total
            print(f"{counter} files processed!")
            break
        counter += 1

    df = pd.DataFrame(
        {
            "filename": pd.Series(files),
            "database": pd.Series(db),
            "syllable": pd.Series(syl),
            "tone": pd.Series(tone),
            "sex": pd.Series(sex),
            "speaker": pd.Series(spkr),
            "sampling_rate": pd.Series(sample),
            "pitch_contour": pd.Series(pitch, dtype=object),
        }
    )
    df.to_json(output)
