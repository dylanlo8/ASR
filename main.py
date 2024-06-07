from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl

def main():
    # Process audio
    processor = Processor()
    
    # Embed audio
    whisper = Whisper()
    train_audio_embeddings, train_labels = whisper.embed_audio(train_dataset)
    
    torch.cuda.empty_cache()

    # Translate embeddings
    translate_model_inst = TranslateModel()

    # Set up the dataset
    df1 = pd.read_csv("csv_1.csv")
    df2 = pd.read_csv("csv_2.csv")

    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Parse through AudioProcessor
    train_dataset = processor.process_audio(concatenated_df['trimmed_segment_path'], concatenated_df['eng_reference'])
    
    # Parse through Whisper Encoder
    train_audio_embeddings, train_transcript = whisper.embed_audio(train_dataset)

    # Parse through DataLoader
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_transcript)
    train_audioloader = DataLoader(train_audiodataset, batch_size=4, shuffle=True)


    lightning_translator = LightningTranslator()

    trainer = pl.Trainer(devices = 4, accelerator= 'auto')
    trainer.fit(model=lightning_translator, train_dataloaders=train_audioloader)

if __name__ == "__main__":
    main()
