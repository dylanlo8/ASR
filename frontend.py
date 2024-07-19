import gradio as gr
import soundfile as sf
import requests
from scipy.io.wavfile import write
import os
from dotenv import load_dotenv

load_dotenv()

def predict(audio):
    sample_rate, audio_data = audio
    temp_file = "temp_audio.wav"
    write(temp_file, sample_rate, audio_data)

    # Send the file to the FastAPI server
    with open(temp_file, "rb") as f:
        fastapi_server_ip = os.getenv("FASTAPI_SERVER_IP")
        fastapi_server_port = os.getenv("FASTAPI_SERVER_PORT")
        response = requests.post(f"{fastapi_server_ip}:{fastapi_server_port}/process/", files={"file": f})

    print(response)
    return response.json()["generated_output"][0][0]

# Create the Gradio interface
interface = gr.Interface(
    fn = predict,
    inputs = gr.Audio(sources="upload", type="numpy"), 
    outputs = "text",
    title = "Audio to Text Translator",
    description = "Upload a .wav file and get the English text translation."
)

# Launch the interface
interface.launch()