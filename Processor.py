from datasets import Dataset, Audio
from transformers import AutoProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import pytorch_lightning as pl

class Processor:
    def __init__(self, audio_encoder="./whisper-medium"):
        self.audio_processor = AutoProcessor.from_pretrained(
            audio_encoder, 
            local_files_only=True
        )
    
    def process_audio(self, list_audio_filepaths, labels):
        print("Processing audio files")
        def prepare_dataset(batch):
            audio = batch["audio"]
            features = self.audio_processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_attention_mask=True,
                return_tensors='pt'
            )

            batch["input_features"] = features['input_features'][0]
            batch["attention_mask"] = features["attention_mask"][0]
            return batch
        
        audio_dataset = Dataset.from_dict({
            "audio": list_audio_filepaths,
            "labels": labels
        })

        audio_dataset = audio_dataset.cast_column("audio", Audio()).map(prepare_dataset)

        del self.audio_processor

        # Contains input_features, attention_mask, labels
        return audio_dataset 
    