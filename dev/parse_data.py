import re
import os
import shutil
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import parselmouth


from .load_audio import load_audio_file
from tonami import pitch_process as pp

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

# TODO: THIS SHIT BROKEN, YEET! (load_audio is EMPTY BRO)
def parse_wav_pitch(file_path, library="parselmouth"):
    file_dict = {}
    file_name = str(file_path)[39:]
    file_dict["filename"] = file_name
    sections = file_name.split("_")
    file_dict["speaker"] = sections[0]
    file_dict["syllable"] = sections[1][: len(sections[1]) - 1]
    file_dict["tone"] = sections[1][-1]

    time_series, file_dict["sample"] = load_audio_file(file_path)
    file_dict["pitch_contour"], voiced_flag, voiced_probs = librosa.pyin(
            time_series, fmin=50, fmax=400)
    
    # TODO: make consistent with other parsing function
    # if(library == "parselmouth"):
    #     sound_file = parselmouth.Sound(str(file_path))
    #     file_dict["sample"] = sound_file.get_sampling_frequency()
        
    #     pitch = sound_file.to_pitch()
    #     pitch_values = pitch.selected_array['frequency']
    #     pitch_values[pitch_values==0] = np.nan
        
    #     file_dict["pitch_contour"] = pitch_values
    # else:
    #     time_series, file_dict["sample"] = load_audio_file(file_path)
    #     file_dict["pitch_contour"], voiced_flag, voiced_probs = librosa.pyin(
    #         time_series, fmin=50, fmax=500
    #     )
    return file_dict

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

def rename_user_testing_audio(source_folder: str, out_folder: str, start_time: datetime, end_time:datetime, user_id: str):
    """
    Filename format: ex{num}_{syl}{tone_num}_{year}-{month}_date_{hour}{min}{sec}.mp3
    ex. ex1_fa2_2022-03-02_011655.mp3
    """
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for f in Path(source_folder).rglob("*.mp3"):
        og_file = str(f)
        match = re.search(r'(ex[0-9]+)_([a-z]*[0-9])_(.+?).mp3', og_file)
        parsed_date = match.group(3)
        syl = match.group(2)
        ex_num = match.group(1)
        # https://stackoverflow.com/questions/35231285/python-how-to-split-a-string-by-non-alpha-characters
        # split all non word chars (basically non-alphanumeric) and underscores
        split_datetime = [int(x) for x in re.split(r'[\W_]+', parsed_date)]
        datetime_obj = datetime.datetime(*split_datetime)
        if start_time <= datetime_obj <= end_time:
            counter = 1
            new_filename = f'{syl}_{user_id}_{ex_num}_R{counter}_user-testing.mp3'
            new_filename = str(Path(out_folder, new_filename))
            while os.path.isfile(new_filename):
                counter += 1
                new_filename = f'{syl}_{user_id}_{ex_num}_R{counter}_user-testing.mp3'
                new_filename = str(Path(out_folder, new_filename))
            shutil.copy(og_file, new_filename)

def select_files_random(source_folder, out_folder, num):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    p = Path(source_folder).rglob("*")
    files = [x for x in p if x.is_file()]
    selected = random.choices(files, k=num)
    for f in selected:
        new_filename = str(Path(out_folder, f.name))
        shutil.copy(str(f), new_filename)

def get_sampled_data_info(source_folder):
    all_files = Path(source_folder).rglob("*.mp3")
    speaker_info = dict()
    tone_info = dict()
    for file_name in all_files:
        sections = str(file_name).split("_")
        word = re.split('(\d+)', sections[0])
        syllable = word[0].split("/")[-1]
        tone = word[1]
        speaker = sections[1]

        if speaker in speaker_info:
            speaker_info[speaker] = speaker_info.get(speaker) + 1
        else: 
            speaker_info[speaker] = 1
            
        if tone in tone_info:
            tone_info[tone] = tone_info.get(tone) + 1
        else: 
            tone_info[tone] = 1
            
    return speaker_info, tone_info
    