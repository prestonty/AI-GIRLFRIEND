# To run the AI, you must run this code on Google Colab
I dont want to pay for expensive servers so the AI are ran on Google Colab. The AI being selected is "Echidna 13B (United)" (WARNING - slightly unfiltered...)
https://colab.research.google.com/github/koboldai/KoboldAI-Client/blob/main/colab/GPU.ipynb#scrollTo=lVftocpwCoYw

# Paste the link from the WEB GUI
Paste the link from the Web GUI into mic_vad_streaming.py into the variable named "ENDPOINT"
E.g. It looks like https://crude-nevertheless-seconds-experts.trycloudflare.com/

# To run the program, run Main.py.
python main.py



# TODO list:
1. Keep a transcript of all the things the AI said
2. Have a filter mode (another program will decide if what the AI is about to say is appropriate). This is a a barrier from KoboldAI and "elevenlabs TTS"
3. Convert the voice from TTS to kafka or someone else (who matches ig). make the tts voice compatible (have different versions). I want: dommy, loli, normal female news reporter type voice, male voice.
4. Make an interface (Vtube studio??)
5. Implement a deafen feature so the AI doesnt pick up your reactions to its input (I NEED THIS)