from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import re
import time

torch.set_float32_matmul_precision('medium')

def clean_text(text):
    # Function to clean individual text
    def clean_string(s):
        # Remove dashes and leading/trailing spaces
        s = re.sub(r'-+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    # If input is a list of strings
    if isinstance(text, list):
        return [clean_string(t) for t in text]
    # If input is a single string
    elif isinstance(text, str):
        return clean_string(text)
    else:
        raise TypeError("Input should be a string or a list of strings.")
  
def main():
    # Set up the dataset
    df1 = pd.read_csv("csv_1.csv").head(250)
    df2 = pd.read_csv("csv_2.csv").head(0)

    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    cleaned_eng_ref = clean_text(concatenated_df['eng_reference'].tolist())

    # Parse through AudioProcessor
    processor = Processor()
    train_dataset = processor.process_audio(concatenated_df['trimmed_segment_path'], cleaned_eng_ref)
    
    # Parse through Whisper Encoder
    del processor
    torch.cuda.empty_cache()

    whisper = Whisper()
    train_audio_embeddings, train_transcript = whisper.embed_audio(train_dataset)

    # Parse through DataLoader
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_transcript)
    train_audioloader = DataLoader(train_audiodataset, batch_size=1, shuffle=False, num_workers=63)

    del whisper
    torch.cuda.empty_cache()

    lightning_translator = LightningTranslator()

    logger = CSVLogger("logs", name="my_exp_name")

    trainer = pl.Trainer(
        devices = 1, 
        accelerator= 'auto',
        max_epochs = 8,
        enable_checkpointing=False,
        logger = logger,
        accumulate_grad_batches=4
    )

    trainer.fit(model=lightning_translator, 
        train_dataloaders=train_audioloader
    )

    trainer.save_checkpoint("checkpoints/first_checkpoint.ckpt")

if __name__ == "__main__":
    main()


