from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
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
    df1 = pd.read_csv("csv_1.csv")
    df2 = pd.read_csv("csv_2.csv")
    df3 = pd.read_csv("csv_3.csv")

    concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)
    cleaned_eng_ref = clean_text(concatenated_df['eng_reference'].tolist())

    # STEP 1: Parse through AudioProcessor
    processor = Processor()
    train_dataset = processor.process_audio(concatenated_df['trimmed_segment_path'], cleaned_eng_ref)
    
    # STEP 2: Parse through Whisper Encoder
    del processor
    torch.cuda.empty_cache()
    whisper = Whisper()
    train_audio_embeddings, train_transcript = whisper.embed_audio(train_dataset)

    # STEP 3: Parse through DataLoader
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_transcript)
    train_audioloader = DataLoader(train_audiodataset, batch_size=1, shuffle=False, num_workers=63)

    del whisper
    torch.cuda.empty_cache()

    # STEP 4: Set up Pytorch Lightning Orchestrator
    lightning_translator = LightningTranslator()

    logger = CSVLogger("logs", name = "my_exp_name")
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = 10,
        dirpath="my_checkpoints/",
        filename="checkpoint",
    )

    trainer = pl.Trainer(
        devices = 1, 
        accelerator = 'auto',
        max_epochs = 30,
        enable_checkpointing = True,
        logger = logger,
        callbacks= [lr_monitor, checkpoint_callback],
        accumulate_grad_batches = 4
    )

    # STEP 5: Train Model
    trainer.fit(model=lightning_translator, 
        train_dataloaders=train_audioloader
    )

    trainer.save_checkpoint("my_checkpoints/final_checkpoint.ckpt")

if __name__ == "__main__":
    main()


