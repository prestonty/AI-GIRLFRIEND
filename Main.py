# python mic_vad_streaming.py
# -m "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.pbmm"
# -s "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.scorer"

# libraries for taking key input
from pynput.keyboard import Key, Listener

# library for running mic_vad_streaming python script
import os
# env
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI

# .env
load_dotenv()

# Keyboard inputs start ------------------------------------------------------

isListening = False

def startListening():
    # code to start listening
    print("Pluto-chan started listening")
    os.system('python mic_vad_streaming/mic_vad_streaming.py -m "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.pbmm" -s "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.scorer"')

def stopListening():
    # code to stop listening
    print("Pluto-chan stopped listening")

    # ERM i cant turn it off.... whelp uwu
    exit()

def on_press(key):
    print('{0} pressed'.format(
        key))
    if key == Key.up: # TODO dont wanna deal with this now
        isListening = True
        startListening()
    elif key == Key.down: # TODO dont wanna deal with this now
        isListening = False
        stopListening()

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# Keyboard inputs end ------------------------------------------------------

# Open AI ------------------------------------------------------------------

os.getenv('OPENAI_KEY')

client = OpenAI(
  organization='org-tv9e9aYfa1J1J0ztFs74fF0r',
)

if __name__ == '__main__':

    # action listener, if l is pressed, turn isListening one

    if(isListening == True):
        print("Pluto is listen to your ass")
