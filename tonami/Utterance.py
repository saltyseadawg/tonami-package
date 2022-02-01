import librosa
# stub for utterance class
# idea right now is that it creates an utterance that has gone through all
# the pre-processing and can be passed to our classifier or visualization model
# possibly stores its own classification/probability as well?

class Utterance:
    def __init__(self, track, sr, fmin, fmax):
        self.track = track
        self.sr = sr

        #we might want to just pass the pitch contour and stuff in directly as constructor? unsure
        #depends on how rest of code is written i guess
        self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(track, fmin, fmax)

    def process(self):
        #calls pitch_process functions
        pass
