from pathlib import Path
from tonami import Classifier as c
from tonami import Utterance as u
from tonami import user
from dev import parse_data as dd
import numpy as np
import pandas as pd
import os
import glob


PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'
SPEAKER_INFO_FILEPATH = 'tonami/data/speaker_max_min.txt'
USER_FILEPATH = 'data/users'
TONE_PERFECT_FILEPATH = 'data/tone_perfect'

def main():
    clf = c.Classifier(4)
    clf.load_clf('tonami/data/pickled_svm_80.pkl')    

    df = pd.read_csv('data/ratings/ratings_all.csv')
    df = df.reset_index()
    files = []
    tone1 = []
    tone2 = []
    tone3 = []
    tone4 = []
    for index, row in df.iterrows():
        file = row['File']
        is_native = row['Native']

        if is_native == 'Y':
            file_mp3 = f'{TONE_PERFECT_FILEPATH}/{file}_MP3.mp3'
            sections = file.split("_")
            speaker = sections[1]
            speakers_info = pd.read_json(SPEAKER_INFO_FILEPATH)
            speaker_max_f0 = speakers_info.loc[speakers_info['speaker_name'] == speaker, 'max_f0']
            speaker_min_f0 = speakers_info.loc[speakers_info['speaker_name'] == speaker, 'min_f0']
            speaker_info = user.User(speaker_max_f0, speaker_min_f0)

            word = u.Utterance(filename = file_mp3)
            if(np.isnan(word.pitch_contour).all()):
               continue
            pitch_contour, nans, features = word.pre_process(speaker_info)
            pred, prob = clf.classify_tones(features)
            tone1.append(prob[0][0])
            tone2.append(prob[0][1])
            tone3.append(prob[0][2])
            tone4.append(prob[0][3])
            files.append(file)

        else:
            file_mp3 = glob.glob(f'{USER_FILEPATH}/{file}*.mp3')[0]
            word = u.Utterance(filename = file_mp3)
            speaker = user.User(word.fmax + 50, word.fmin - 50)
            normalized_pitch, nans, features = word.pre_process(speaker)
            if(np.isnan(word.pitch_contour).all()):
                continue
            pred, prob = clf.classify_tones(features)
            tone1.append(prob[0][0])
            tone2.append(prob[0][1])
            tone3.append(prob[0][2])
            tone4.append(prob[0][3])
            files.append(file)

    df_probs = pd.DataFrame({
        'File' : files,
        'Tone1' : tone1,
        'Tone2' : tone2,
        'Tone3' : tone3,
        'Tone4' : tone4
    })
    df_probs.to_csv('ml_rating_probs.csv')
            
    # mtd = []
    # folder_path = "data/l2_corpus"
    # for f in Path(folder_path).rglob("*.*"):
    #     # mtd.append(dd.parse_wav_pitch(f)) #TODO: this is so inefficent lol

    #     word = u.Utterance(filename=f, pitch_filepath=None) #above and below may have diff pitch contours (due to hardcoded fmin fmax)
    #     # since we are only looking at one track, gives the range a bit of wiggle room
    #     word.fmin -= 50
    #     word.fmax += 50
    #     speaker = user.User(word.fmax, word.fmin)
    #     normalized_pitch, nans, features = word.pre_process(speaker)
        
    #     if(np.isnan(word.pitch_contour).all()):
    #         continue

    #     pred, prob = svm.classify_tones(features)
    #     print(str(f) + ", " + str(pred) + ", " + str(np.around(prob * 100, decimals=1)) ) #TODO: make fstring

main()