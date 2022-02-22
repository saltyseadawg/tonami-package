# stub for user class
# holds any and all user relevant data we need

class User:
    def __init__(self, max_f0=400, min_f0=50):
        self.pitch_profile["max_f0"] = max_f0
        self.pitch_profile["min_f0"] = min_f0

    def get_pitch_profile(self) -> dict:
        return self.pitch_profile

    def set_max_pitch(self, max_f0):
        self.pitch_profile["max_f0"] = max_f0

    def set_min_pitch(self, min_f0):
        self.pitch_profile["min_f0"] = min_f0
