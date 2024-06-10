from transformers import WhisperModel
from torch.utils.data import DataLoader
import torch

class Whisper:
    """
    Class for embedding audio inputs using a pre-trained Whisper model.
    
    Attributes:
        device_type (torch.device): Device type (CUDA if available, else CPU).
        audio_encoder (WhisperModel): Pre-trained Whisper model for audio encoding.
    """
    
    def __init__(self, audio_encoder="./whisper-medium"):
        """
        Initializes the Whisper class with the specified pre-trained Whisper model.
        
        Args:
            audio_encoder (str): Path to the pre-trained Whisper model.
        """
        self.BATCH_SIZE = 32
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Whisper")
        self.audio_encoder = WhisperModel.from_pretrained(audio_encoder, local_files_only=True).to(self.device_type)
        print("Whisper Loaded and ready to embed audio inputs")
        
    def embed_audio(self, audio_dataset):
        """
        Embeds the audio inputs from the dataset using the Whisper model.
        
        Args:
            audio_dataset (Dataset): Dataset containing audio features and attention masks.
        
        Returns:
            tuple: Tuple containing audio embeddings and labels.
        """
        # Batch Embedding

        print(audio_dataset)

        audio_dataset.set_format(type='torch', columns=['input_features', 'attention_mask', 'labels'])
        data_loader = DataLoader(audio_dataset, batch_size = self.BATCH_SIZE, shuffle=False)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # Push to cuda
                inputs = batch['input_features'].to(self.device_type)
                att_mask = batch['attention_mask'].to(self.device_type)
                labels = batch['labels']

                # Embed Audio Batch
                encoder_outputs = self.audio_encoder.encoder(
                    inputs,
                    output_hidden_states=True,
                    attention_mask=att_mask
                )

                # Extract audio embeddings
                audio_embeddings = encoder_outputs.last_hidden_state.to('cpu')

                all_embeddings.append(audio_embeddings)
                all_labels.extend(labels)

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Clear memory
        del self.audio_encoder
        torch.cuda.empty_cache()

        # To be parsed into Dataloader
        return all_embeddings, all_labels 
