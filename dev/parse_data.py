from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import parselmouth
import pickle
# import sklearn
# import sklearn.pipeline

from .load_audio import load_audio_file
from tonami import pitch_process as pp

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

def parse_toneperfect_pitch(file_path, library):
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
    if(library == "parselmouth"):
        sound_file = parselmouth.Sound(str(file_path))
        file_dict["sample"] = sound_file.get_sampling_frequency()
        
        pitch = sound_file.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        
        file_dict["pitch_contour"] = pitch_values
    else:
        time_series, file_dict["sample"] = load_audio_file(file_path)
        file_dict["pitch_contour"], voiced_flag, voiced_probs = librosa.pyin(
            time_series, fmin=50, fmax=500
        )
    return file_dict


def write_toneperfect_pitch_data(
    folder_path="data/tone_perfect/tone_perfect_all_mp3",
    output="data/parsed/toneperfect_pitch.json",
    library="librosa"
):
    """Creates a json file containing metadata and pitch information from
    Tone Perfect files found in folder_path.

    Args:
        folder_path (str): target folder that is searched
        output (str): name of file that is created
        library (str): name of library to be used for pitch extraction
    """

    files, db, syl, tone, sex, spkr, sample, pitch = ([] for i in range(8))
    counter = 0
    for f in Path(folder_path).rglob("*.mp3"):
        mtd = parse_toneperfect_pitch(f, library)
        files.append(mtd["filename"])
        db.append(mtd["database"])
        syl.append(mtd["syllable"])
        tone.append(mtd["tone"])
        sex.append(mtd["sex"])
        spkr.append(mtd["speaker"])
        sample.append(mtd["sample"])
        pitch.append(mtd["pitch_contour"])
        if counter%100 == 0:
            # 9840 Tone Perfect files total
            print(f"{counter} files processed!")
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

def save_speaker_max_min():
    """Creates a json file containing max and min f0 frequencies from the 
    6 test speakers from Tone Perfect Database.
    """

    output="tonami/data/speaker_max_min.txt"

    pitch_data = pd.read_json(PITCH_FILEPATH)
    speakers = ['FV1', 'FV2', 'FV3', 'MV1', 'MV2', 'MV3']
    spkr_max, spkr_min = [], []

    for i in range(len(speakers)):
        # Get raw tracks for each speaker        
        spkr_data = pitch_data.loc[pitch_data['speaker'] == speakers[i]]

        # Preprocessing
        _, data_valid = pp.preprocess_all(spkr_data)

        # Get min and max
        max_f0, min_f0 = pp.max_min_f0(data_valid)

        # Add to array
        spkr_max.append(max_f0)
        spkr_min.append(min_f0)
    
    df = pd.DataFrame(
        {
            "speaker_name": pd.Series(speakers),
            "max_f0": pd.Series(spkr_max),
            "min_f0": pd.Series(spkr_min),
        }
    )
    df.to_json(output)

def save_classifier_data(
    clf,
    name = "svm_80"
):
    file_name = "tonami/data/pickled_" + name + ".pkl"

    # pitch_data = pd.read_json(PITCH_FILEPATH)
    # label, data = pp.end_to_end(pitch_data)
    
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, test_size=0.9)

    # clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(gamma='auto'))
    # clf.fit(X_train, y_train)

    pickle.dump(clf, open(file_name, 'wb'))