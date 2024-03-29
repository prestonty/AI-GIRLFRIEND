To run, paste this into the terminal:

python mic_vad_streaming.py -m "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.pbmm" -s "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.scorer"

Optimization:
- TTS is very fast
- STT is pretty decent. it doesnt pick everything up though. (E.g. first few words).

TODO list:
1. keep a transcript of all the things the AI said
2. have a filter mode (another program will decide if what the AI is about to say is appropriate). This is a a barrier from "receiving the text from gpt" and "elevenlabs TTS"
3. convert the voice from TTS to kafka or someone else (who matches ig). make the tts voice compatible (have different versions). I want: dommy, loli, normal female news reporter type voice, male voice.

# To run the AI, you must run this code on Google Colab. I dont want to pay for expensive servers so the AI are ran on Google Colab
The AI being selected is "Echidna 13B (United)" (slightly unfiltered and )
https://colab.research.google.com/github/koboldai/KoboldAI-Client/blob/main/colab/GPU.ipynb#scrollTo=lVftocpwCoYw
