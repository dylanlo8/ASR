from datasets import Dataset, Audio
from transformers import AutoProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import pytorch_lightning as pl

class Processor:
    """
    Class for processing audio files and preparing them for input into a model.
    
    Attributes:
        audio_processor (AutoProcessor): Processor for extracting features from audio files.
    """
    
    def __init__(self, audio_encoder="./whisper-medium"):
        """
        Initializes the Processor with the specified pre-trained audio encoder.
        
        Args:
            audio_encoder (str): Path to the pre-trained audio encoder.
        """
        self.audio_processor = AutoProcessor.from_pretrained(
            audio_encoder, 
            local_files_only=True
        )
    
    def process_audio(self, list_audio_filepaths, labels):
        """
        Processes a list of audio file paths and their corresponding labels to extract features and attention masks.
        
        Args:
            list_audio_filepaths (list): List of audio file paths.
            labels (list): List of labels corresponding to the audio files.
        
        Returns:
            Dataset: Dataset containing input features, attention masks, and labels.
        """
        print("Processing audio files")

        def prepare_dataset(batch):
            """
            Prepares the dataset by extracting features and attention masks from the audio files.
            
            Args:
                batch (dict): Batch of data containing audio information.
            
            Returns:
                dict: Batch with extracted input features and attention masks.
            """
            audio = batch["audio"]
            features = self.audio_processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_attention_mask=True,
                return_tensors='pt'
            )

            batch["input_features"] = features['input_features'][0]
            batch["attention_mask"] = features["attention_mask"][0]
            batch["labels"] = batch["labels"] + " <|endoftext|>"
            return batch
        
        audio_dataset = Dataset.from_dict({
            "audio": list_audio_filepaths,
            "labels": labels
        })

        audio_dataset = audio_dataset.cast_column("audio", Audio()).map(prepare_dataset)

        del self.audio_processor

        # Contains input_features, attention_mask, labels
        return audio_dataset