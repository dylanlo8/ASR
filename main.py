import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from Whisper import Whisper
from Processor import Processor
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import os

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Set up the dataset
    df = pd.read_csv('combined_train_clean.csv').head(1)

    # STEP 1: Parse through AudioProcessor
    processor = Processor()
    train_dataset = processor.process_audio(df['trimmed_segment_path'], df['eng_reference'])
    
    # STEP 2: Parse through Whisper Encoder
    del processor
    torch.cuda.empty_cache()

    whisper = Whisper()
    train_audio_embeddings, train_transcript = whisper.embed_audio(train_dataset)

    # STEP 3: Parse through DataLoader
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_transcript)
    train_audioloader = DataLoader(train_audiodataset, batch_size=1, shuffle=True, num_workers=63)

    del whisper
    torch.cuda.empty_cache()

    # STEP 4: Set up Orchestrator
    lightning_translator = LightningTranslator()

    # Load pretrained adaptor weights (If needed)
    # checkpoint = torch.load("checkpoints/with_lr_scheduler.ckpt")
    # adaptor_weights = {k[len("model.adaptor."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.adaptor")}
    # lightning_translator.model.adaptor.load_state_dict(adaptor_weights)

    # Initialise extra training params
    logger = CSVLogger("logs", name = "my_exp_name")
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = 3,
        dirpath="my_checkpoints/",
        filename="checkpoint_llama_{epoch}",
    )

    trainer = pl.Trainer(
        devices = 1,
        accelerator = 'auto',
        max_epochs = 6,
        enable_checkpointing = True,
        logger = logger,
        callbacks= [lr_monitor, checkpoint_callback],
        accumulate_grad_batches = 4,
    )

    # STEP 5: Train Model
    trainer.fit(
        model=lightning_translator, 
        train_dataloaders=train_audioloader,
        ckpt_path = 'my_checkpoints/checkpoint_llama_epoch=5.ckpt',
    )

    trainer.save_checkpoint("checkpoints/epoch_llama=6.ckpt")

if __name__ == "__main__":
    main()