import gradio as gr
import soundfile as sf
import requests
from scipy.io.wavfile import write
import os
import numpy as np
from dotenv import load_dotenv
from vad import VADIterator
import librosa

load_dotenv()
fastapi_server_ip = os.getenv("FASTAPI_SERVER_IP")
fastapi_server_port = os.getenv("FASTAPI_SERVER_PORT")

WINDOW_SIZE = 8000 # 0.5 sec
SAMPLE_RATE = 16000

def predict(filepath):
    audio_data, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
    temp_file = "temp_audio.wav"

    # Intialise VAD
    vad_iterator = VADIterator()

    accumulated_chunks = np.array([])
    accumulated_results = ""

    for i in range(0, len(audio_data), WINDOW_SIZE):
        chunk = audio_data[i:i+WINDOW_SIZE]

        # post to VAD
        response = vad_iterator(chunk)

        # still a voice, continue to process next chunk
        if response is None:
            accumulated_chunks = np.append(accumulated_chunks, chunk)
            continue
        
        # start of human voice activity
        if response.get('start') is not None:
            # Start accumulation
            accumulated_chunks = chunk

        # end of human voice activity
        if response.get('end') is not None:
            # save as temp wav file
            write(temp_file, SAMPLE_RATE, accumulated_chunks)

            # reset accumulated chunks
            accumulated_chunks = np.array([])

            # accumualte results
            accumulated_results = accumulated_results + "\n" + (post_audio_request(temp_file))
            
            # pass results to gradio frontend
            yield accumulated_results

    # if there are chunks left
    if len(accumulated_chunks) > 0:
        write(temp_file, SAMPLE_RATE, accumulated_chunks)
        accumulated_results = accumulated_results + "\n" + (post_audio_request(temp_file))
        yield accumulated_results

def post_audio_request(temp_file):
    # send to endpoint for transcription
    with open(temp_file, "rb") as f:
        response = requests.post(f"{fastapi_server_ip}:{fastapi_server_port}/process/", files={"file": f})
        result = response.json()["generated_output"][0][0]
        return result

# Create the Gradio interface
interface = gr.Interface(
    fn = predict,
    inputs = gr.Audio(sources="upload", type="filepath"),
    outputs = "text",
    title = "Audio to Text Translator",
    description = "Upload a .wav file and get the English text translation.",
    live = True
)

# Launch the interface
interface.launch()