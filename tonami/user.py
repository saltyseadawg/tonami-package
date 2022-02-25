# stub for user class
# holds any and all user relevant data we need
# import logging

class User:
    def __init__(self, max_f0=50, min_f0=400):

        self.pitch_profile = {
            "max_f0": max_f0,
            "min_f0": min_f0
        }

    def get_pitch_profile(self) -> dict:
        return self.pitch_profile

    # def calibrate_pitch(self, audio: AudioSegment, floor=50, ceil=400):
    #     """Sets the user's min and max pitch based on the input audio.
    #     """
    #     # librosa can't load mp3 buffer? -> use wav format
    #     wav = audio_utils.convert_audio(audio, 'wav')
    #     y, sr = librosa.load(wav)

    #     # need to figure out how to adjust the floor and ceiling at will
    #     pitch_contour, _, _ = librosa.pyin(y, fmin=floor, fmax=ceil)
    #     # logging.info(pitch_contour)
    #     # pitch_contour = pp.preprocess(pitch_contour)
    #     max_f0, min_f0 = pp.max_min_f0(pitch_contour)

    #     # adjust max and min since user likely not actually hitting their max and mins? trying to avoid negative values when normalizing
    #     self.pitch_profile['max_f0'] = max_f0 + 20
    #     self.pitch_profile['min_f0'] = min_f0 - 20

