#loads desired audio from either files or parsed data
#these guys could return a list of utterances or tracks? i'm not sure yet.
#depends on rest of code?
import warnings

# known error of package, we intend to use audioread.
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)


def load_audio_file(filepath):
    #loads a single, specific file?
    pass

def load_audio_parsed(syllable, spkr, sex, tone, database):
    #loads relevant things by reading in our nice parsed text file database
    pass

