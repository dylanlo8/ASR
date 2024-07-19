from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
from torch.utils.data import DataLoader
from Whisper import Whisper
from Processor import Processor
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import pytorch_lightning as pl
import uvicorn
import numpy as np
from scipy.io import wavfile
import traceback

torch.set_float32_matmul_precision('medium')

app = FastAPI()
processor = Processor()
whisper = Whisper()
model = LightningTranslator.load_from_checkpoint(checkpoint_path="checkpoints/with_lr_scheduler_and_cleandata.ckpt").to("cuda")

@app.post("/process/")
async def process_audio(file : UploadFile = File(...)):
    try:
        audio_path = file.filename
        with open(audio_path, "wb+") as fp:
            fp.write(file.file.read())
            
        #take read audio wav numpy array and store it as temporary file
        # wavfile.write('temp.wav', 16000, audio_request.audio_array)

        # Step 1: Process audio file path inputs
        audio_dataset = processor.process_audio(list_audio_filepaths= [audio_path], labels = [''])
        
        # Step 2: Embed audio input features
        audio_embeddings, labels = whisper.embed_audio(audio_dataset)

        # Step 3: Set up Data Loader to stream audio inputs for inference
        embed_dataset = AudioEmbeddingsDataset(audio_embeddings, labels)
        embed_dataloader = DataLoader(embed_dataset, batch_size=1, shuffle=False, num_workers=4)

        results = []
        for batch_idx, batch in enumerate(embed_dataloader):
            tokenised_output = model.predict_step(batch, batch_idx)
            results.append(model.decode(tokenised_output))

        return {"generated_output": results}
    
    except Exception as e:
        print(e)
        error_trace = traceback.format_exc()
        print(error_trace)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
