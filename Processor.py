from datasets import Dataset, Audio
from transformers import AutoProcessor

class Processor:
    """
    Class for processing Audio .wav files and preparing them as input features into Whisper's Automatic
    Speech Recognition Model.
    
    Attributes:
        audio_processor (AutoProcessor): Processor for extracting features from audio files.
    """
    
    def __init__(self, path_to_whisper_model='./whisper-medium'):
        self.audio_processor = AutoProcessor.from_pretrained(
            path_to_whisper_model, 
            local_files_only=True
        )
    
    def process_audio(self, list_audio_filepaths, labels=None):
        """
        Processes a list of audio file paths and their corresponding labels to extract features and attention masks.
        
        Args:
            list_audio_filepaths (list): List of audio file paths.
            labels (list or None): List of labels corresponding to the audio files. Defaults to None.
        
        Returns:
            audio_dataset (Dataset): Dataset containing input_features, attention masks, and labels (if provided).
                - input_features (torch.Tensor): Tensor containing extracted input features from the audio files.
                - attention_mask (torch.Tensor): Tensor containing attention masks for the input features.
                - labels (list of str, optional): List of labels corresponding to the audio files, if provided.
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

        # Cast and process
        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000)).map(prepare_dataset)

        # Contains input_features, attention_mask, labels (if provided)
        return audio_dataset
