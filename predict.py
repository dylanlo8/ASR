from Whisper import Whisper
from Processor import Processor
from Orchestrator import AudioEmbeddingsDataset, LightningTranslator
import torch
from torch.utils.data import DataLoader
import pandas as pd

torch.set_float32_matmul_precision('medium')
  
def main():
    # Read test  dataset
    #test_df = pd.read_csv("data/sub/hi.csv").head(30)

    # STEP 1: Parse through AudioProcessor
    processor = Processor()
    # test_dataset = processor.process_audio(test_df['trimmed_segment_path'], test_df['eng_reference'])
    test_dataset = processor.process_audio(['lifelong_learning_data.wav'])

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