# AI Girlfriend 💖

AI Girlfriend is an experimental project that lets you talk to a highly personalized AI companion with natural natural-sounding voice and emotion. I personally fine-tuned the voice! This isn't just a chatbot — it's a custom character brought to life that's the cure to male loneliness, created years ahead of Grok.

---

## 🧰 Features

- Natural conversation flow using KoboldAI's Echidna 13B model (slightly unfiltered... you’ve been warned)
- Real-time voice interaction with ElevenLabs TTS
- Easy setup with Google Colab for free GPU access
- Lightweight — no server hosting needed
- Fully customizable prompt and voice personality

---

## ⚙️ Setup Instructions

#### 1. Create Virtual Environment & Activate It

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```
#### 2. Install Dependencies
`pip install -r requirements.txt`

---

### ☁️ Run KoboldAI on Google Colab
Can't pay for expensive servers? Run the AI in your browser with free GPUs:

#### 🔗 Launch KoboldAI on Google Colab

Select "Echidna 13B (United)"
⚠️ Slightly unfiltered — use with caution

Wait ~10–25 minutes for the model to load.

---

### 🔗 Set the Web GUI Endpoint
After the model loads, KoboldAI will give you a public URL like:

```bash
# Example output URL from Colab
https://crude-nevertheless-seconds-experts.trycloudflare.com/

# Paste this link into the ENDPOINT variable inside:
mic_vad_streaming.py
```

---

### 🔐 Add Your ElevenLabs API Key
Create a file named .env in the root directory

Add the following line (replace with your key):
`ELEVEN_LABS_KEY=your_key_here`
Create an account at https://elevenlabs.io if you don’t have one.

---

### ▶️ Run the Program
```bash
python main.py
```

---

### 🧠 TODO List
 Keep a transcript of all the things the AI says

 Add a filter mode (prevents raw KoboldAI output from going directly into ElevenLabs)

 Build a user interface (maybe integrate with VTube Studio?)

 Implement a “deafen” toggle so the AI ignores your background noise or reactions

