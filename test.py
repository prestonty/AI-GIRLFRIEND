# This is for testing the eleven labs API

import requests
import os
import audioread
import numpy as np
import sounddevice as sd

VOICE_ID = "VkRQXxkfeZ8MQzwDOLue"  # ID of the voice model to use
tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
XI_API_KEY = os.getenv('ELEVEN_LABS_KEY')  # Your API key for authentication
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time

headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY
}

data = {
    "text": "hi", # Change the message here
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,  # change these parameters as needed
        "use_speaker_boost": True
    }
}

response = requests.post(tts_url, headers=headers, json=data, stream=True)

# Define a callback function to handle audio playback
def audio_callback(outdata, frames, time, status):
    global audio_data
    # Read audio data from the streaming response
    chunk = response.raw.read(CHUNK_SIZE)
    if len(chunk) == 0:
        print("End of audio stream")
        raise sd.CallbackAbort
    audio_data = np.frombuffer(chunk, dtype=np.int16).reshape((-1, 2))
    outdata[:] = audio_data

# Open a stream for audio playback
with sd.OutputStream(channels=2, callback=audio_callback):
    print("Playing audio...")
    # Keep the script running while audio is being played
    while True:
        pass
