from pathlib import Path
from tonami import Classifier as c
from tonami import Utterance as u
from tonami import user
from dev import parse_data as dd
import numpy as np

def main():
    svm = c.Classifier(4)
    # mtd = []
    folder_path = "data/l2_corpus"
    for f in Path(folder_path).rglob("*.*"):
        # mtd.append(dd.parse_wav_pitch(f)) #TODO: this is so inefficent lol

        word = u.Utterance(filename=f, pitch_filepath=None) #above and below may have diff pitch contours (due to hardcoded fmin fmax)
        # since we are only looking at one track, gives the range a bit of wiggle room
        word.fmin -= 50
        word.fmax += 50
        speaker = user.User(word.fmax, word.fmin)
        normalized_pitch, nans, features = word.pre_process(speaker)
        
        if(np.isnan(word.pitch_contour).all()):
            continue

        pred, prob = svm.classify_tones(features)
        print(str(f) + ", " + str(pred) + ", " + str(np.around(prob * 100, decimals=1)) ) #TODO: make fstring

main()