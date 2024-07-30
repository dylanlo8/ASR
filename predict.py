from Whisper import Whisper
from Processor import Processor
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd

torch.set_float32_matmul_precision('medium')
  
def main():
    # Set up the dataset
    # df1 = pd.read_csv("data/sub/hi.csv").head(1)
    # df2 = pd.read_csv("csv_1.csv").head(0)

    # concatenated_df = pd.concat([df1, df2], ignore_index=True)
   
    #cleaned_eng_ref = clean_text(concatenated_df['eng_reference'].tolist())

    # STEP 1: Parse through AudioProcessor
    processor = Processor()
    test_dataset = processor.process_audio(['data/demotest/ai.wav'], [''])
    
    # STEP 2: Parse through Whisper Encoder
    del processor
    torch.cuda.empty_cache()
    whisper = Whisper()
    audio_embeddings, transcript = whisper.embed_audio(test_dataset)

    # STEP 3: Parse through DataLoader
    test_audiodataset = AudioEmbeddingsDataset(audio_embeddings, transcript)
    test_audioloader = DataLoader(test_audiodataset, batch_size=1, shuffle=False, num_workers=63)
    del whisper
    torch.cuda.empty_cache()

    # STEP 4: Load model from checkpoint
    model = LightningTranslator.load_from_checkpoint(checkpoint_path="checkpoints/with_lr_scheduler_and_cleandata.ckpt").to("cuda")

    for batch_idx, batch in enumerate(test_audioloader):
        model.predict_step(batch, batch_idx)

if __name__ == "__main__":
    main()