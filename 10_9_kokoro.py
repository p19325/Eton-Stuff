# pip install kokoro pydub

from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import subprocess
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------- #

title = "Never Gonna Give You Up"

text = """We're no strangers to love, You know the rules and so do I, A full commitment's what I'm thinking of, You wouldn't get this from any other guy.
I just wanna tell you how I'm feeling, Gotta make you understand.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you.
We've known each other for so long, Your heart's been aching, but you're too shy to say it, Inside, we both know what's been going on, We know the game and we're gonna play it.
And if you ask me how I'm feeling, Don't tell me you're too blind to see.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you.
(Ooh, give you up), (Ooh, give you up), Never gonna give, never gonna give, (Give you up), Never gonna give, never gonna give, (Give you up).
We've known each other for so long, Your heart's been aching, but you're too shy to say it, Inside, we both know what's been going on, We know the game and we're gonna play it.
I just wanna tell you how I'm feeling, Gotta make you understand.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you.
Never gonna give you up, Never gonna let you down, Never gonna run around and desert you, Never gonna make you cry, Never gonna say goodbye, Never gonna tell a lie and hurt you."""

# ----------------------------------------- #

pipeline = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')
generator = pipeline(text, voice='bf_isabella')

#pipeline = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')
#generator = pipeline(text, voice='bm_george')

#pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
#generator = pipeline(text, voice='af_heart')

#pipeline = KPipeline(lang_code='h', repo_id='hexgrad/Kokoro-82M')
#generator = pipeline(text, voice='hm_omega')

# ----------------------------------------- #

def play_mp3(path):
    if sys.platform.startswith('darwin'):        
        subprocess.run(['afplay', path], check=True)
    else:
        print("Sorry, playback only implemented for MacOS.")

output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

combined = AudioSegment.empty()

for subchunk_counter, (gs, ps, audio) in enumerate(generator):    
    
    print(gs)

    audio_np = audio.detach().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    segment = AudioSegment(data=audio_int16.tobytes(), sample_width=2, frame_rate=24000, channels=1)
    
    combined += segment
    
mp3_file = f"{output_dir}/{title}.mp3"
combined.export(mp3_file, format="mp3")
play_mp3(mp3_file)
