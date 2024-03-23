# The 'requests' and 'json' libraries are imported. 
# 'requests' is used to send HTTP requests, while 'json' is used for parsing the JSON data that we receive from the API.
import requests
import json
import os

# .env
from dotenv import load_dotenv
load_dotenv()

# Define constants for the script
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
XI_API_KEY = os.getenv('ELEVEN_LABS_KEY')  # Your API key for authentication
VOICE_ID = "VkRQXxkfeZ8MQzwDOLue"  # ID of the voice model to use
TEXT_TO_SPEAK = "I need you in my life"  # Text you want to convert to speech
OUTPUT_PATH = "./audio/output.mp3"  # Path to save the output audio file

# Construct the URL for the Text-to-Speech API request
tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

# Set up headers for the API request, including the API key for authentication
headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY
}

# Set up the data payload for the API request, including the text and voice settings
data = {
    "text": TEXT_TO_SPEAK,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True
    }
}

# Make the POST request to the TTS API with headers and data, enabling streaming response
response = requests.post(tts_url, headers=headers, json=data, stream=True)

# Check if the request was successful
if response.ok:
    # Open the output file in write-binary mode
    with open(OUTPUT_PATH, "wb") as f:
        # Read the response in chunks and write to the file
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    # Inform the user of success
    print("Audio stream saved successfully.")
else:
    # Print the error message if the request was not successful
    print(response.text)