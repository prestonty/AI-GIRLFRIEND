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
<br />
It will take around 10-25 minutes to load the model.

# Paste the link from the WEB GUI
Paste the generated link into the ENDPOINT variable located in mic_vad_streaming.py
<br />
The link will look like https://crude-nevertheless-seconds-experts.trycloudflare.com/

# Add .env file
Create an ElevenLabs Account and copy your API key into your .env file in the variable "ELEVEN_LABS_KEY"

# Run the program
```console
python main.py
```

# TODO list:
1. Keep a transcript of all the things the AI said
2. Have a filter mode. This is a a barrier from KoboldAI and "elevenlabs TTS"
3. Make an interface (Vtube studio??)
4. Implement a deafen feature so the AI doesnt pick up your reactions to its input or miscellaneous noises (audio compression)
