To run, paste this into the terminal:

python mic_vad_streaming.py -m "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.pbmm" -s "C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/deepspeech-0.9.3-models.scorer"

Optimization:
- TTS is very fast
- STT is pretty decent. it doesnt pick everything up though. (E.g. first few words).

TODO list:
1. keep a transcript of all the things the AI said
2. have a filter mode (another program will decide if what the AI is about to say is appropriate). This is a a barrier from "receiving the text from gpt" and "elevenlabs TTS"
3. convert the voice from TTS to kafka or someone else (who matches ig). make the tts voice compatible (have different versions). I want: dommy, loli, normal female news reporter type voice, male voice.
