from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger

torch.set_float32_matmul_precision('medium') 

def main():
    # Set up the dataset
    # df1 = pd.read_csv("csv_1.csv")
    df2 = pd.read_csv("csv_2.csv")

    #concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Parse through AudioProcessor
    processor = Processor()
    train_dataset = processor.process_audio(df2['trimmed_segment_path'], df2['eng_reference'])
    
    # Parse through Whisper Encoder
    torch.cuda.empty_cache()
    whisper = Whisper()
    train_audio_embeddings, train_transcript = whisper.embed_audio(train_dataset)

    # Parse through DataLoader
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_transcript)
    train_audioloader = DataLoader(train_audiodataset, batch_size=8, shuffle=False, num_workers=63)

    torch.cuda.empty_cache()
    lightning_translator = LightningTranslator()

    logger = CSVLogger("logs", name="my_exp_name")

    trainer = pl.Trainer(devices = 1, 
        accelerator= 'auto', 
        accumulate_grad_batches = 4, 
        enable_checkpointing=False, 
        logger = logger
    )

    trainer.fit(model=lightning_translator, 
        train_dataloaders=train_audioloader
    )

if __name__ == "__main__":
    main()
