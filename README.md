# Create virtual environment and activate it
```console
python -m venv .venv
.venv/Scripts/activate
```

# Install Dependencies
```console
pip install -r requirements.txt
```

# Run KoboldAI on Google Colab
Can't pay for expensive servers so the AI is executed on [Google Colab](https://colab.research.google.com/github/koboldai/KoboldAI-Client/blob/main/colab/GPU.ipynb#scrollTo=lVftocpwCoYw)
<br />
Use "Echidna 13B (United)" (WARNING - slightly unfiltered...)

# Paste the link from the WEB GUI
Paste the generated link into the ENDPOINT variable located in mic_vad_streaming.py
<br />
The link will look like https://crude-nevertheless-seconds-experts.trycloudflare.com/

# Run the program
```console
python main.py
```

# TODO list:
1. Keep a transcript of all the things the AI said
2. Have a filter mode (another program will decide if what the AI is about to say is appropriate). This is a a barrier from KoboldAI and "elevenlabs TTS"
3. Convert the voice from TTS to kafka or someone else (who matches ig). make the tts voice compatible (have different versions). I want: dommy, loli, normal female news reporter type voice, male voice.
4. Make an interface (Vtube studio??)
5. Implement a deafen feature so the AI doesnt pick up your reactions to its input (I NEED THIS)