from pathlib import Path

def parse_toneperfect():
    file1 = open("utils/tonePerfect.txt", "w")
    for file in Path("data/tone_perfect/tone_perfect_all_mp3").rglob("*.mp3"):
        file_name = str(file)[39:]
        sections = file_name.split("_")
        sex = sections[1][0]
        speaker = sections[1]
        syllable = sections[0][:len(sections[0])-1]
        tone = sections[0][-1]
        db = "toneperfect"
        file1.write(file_name + ", " + sex + ", " + speaker + ", " + syllable + ", " + tone + ", " + db + "\n")
    file1.close()
parse_toneperfect()