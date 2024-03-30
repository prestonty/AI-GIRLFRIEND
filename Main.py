import subprocess

# RUN THIS FILE TO START THE PROGRAM INSTEAD OF RUNNING THE COMMAND
def run_mic_vad_streaming():
    command = [
        "python",
        "./mic_vad_streaming/mic_vad_streaming.py",
        "-m",
        "./models/deepspeech-0.9.3-models.pbmm",
        "-s",
        "./models/deepspeech-0.9.3-models.scorer"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    except FileNotFoundError:
        print("Error: /mic_vad_streaming/mic_vad_streaming.py script not found")

if __name__ == "__main__":
    run_mic_vad_streaming()