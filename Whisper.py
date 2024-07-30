from transformers import WhisperModel
from torch.utils.data import DataLoader
import torch
import config

class Whisper:
    """
    This class provides methods to load a pre-trained Whisper Model and use it to generate embedded representations of the audio data.
    The Whisper's embedded representations have the dimensions (batch_size, sequence_length, embedding_dimensions) which are fixed to (X, 1500, 1024).

    Attributes:
        BATCH_SIZE (int): Batch size for data loader.
        device_type (torch.device): Device to run the model on (CUDA or CPU).
        audio_encoder (WhisperModel): Pre-trained Whisper model for audio encoding.
    """
    
    def __init__(self, path_to_whisper_model = config.ASR['ASR_NAME']):
        self.BATCH_SIZE = 32
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Whisper")
        self.audio_encoder = WhisperModel.from_pretrained(
            path_to_whisper_model,
            local_files_only=True
        ).to(self.device_type)
        
        print("Whisper Loaded and ready to embed audio inputs")
        
    def embed_audio(self, audio_dataset):
        """
        Embeds the audio inputs from the dataset using the Whisper's encoder.
        
        Args:
            audio_dataset (Dataset): Dataset containing input_features, attention masks, and labels.
                - input_features (torch.Tensor): Tensor containing extracted input features from the audio files.
                - attention_mask (torch.Tensor): Tensor containing attention masks for the input features.
                - labels (list of str): List of labels corresponding to the audio files.
        
        Returns:
            all_embeddings (torch.Tensor): Audio embeddings tensor.
            all_labels (list of str): List of all the labels.
        """
        audio_dataset.set_format(type='torch', columns=['input_features', 'attention_mask', 'labels'])

        # Initialize DataLoader for batch encoding
        data_loader = DataLoader(audio_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch['input_features'].to(self.device_type)
                att_mask = batch['attention_mask'].to(self.device_type)
                labels = batch['labels']
                all_labels.extend(labels)

                # Batch forward pass through Whisper's Encoder 
                encoder_outputs = self.audio_encoder.encoder(
                    inputs,
                    output_hidden_states=True,
                    attention_mask=att_mask
                )

                # Retrieve the encoded audio representations from the last hidden layer from the Encoder
                audio_embeddings = encoder_outputs.last_hidden_state.to('cpu')
                all_embeddings.append(audio_embeddings)

        # Consolidate all embeddings into one tensor
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings, all_labels