from transformers import WhisperModel
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
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Whisper")
        self.audio_encoder = WhisperModel.from_pretrained(audio_encoder, local_files_only=True).to(self.device_type)
        print("Whisper Loaded and ready to embed audio inputs")
        
    def embed_audio(self, audio_dataset):
        """
        Embeds the audio inputs from the dataset using the Whisper model.
        
        Args:
            audio_dataset (dict): Dataset containing audio features and attention masks.
        
        Returns:
            tuple: Tuple containing audio embeddings and labels.
        """
        inputs = torch.tensor(audio_dataset['input_features']).to(self.device_type)
        att_mask = torch.tensor(audio_dataset['attention_mask']).to(self.device_type)
        labels = torch.tensor(audio_dataset['labels']).to(self.device_type)

        with torch.no_grad():
            encoder_outputs = self.audio_encoder.encoder(
                inputs, 
                output_hidden_states=True,
                attention_mask=att_mask
            )

        # Clear memory
        del self.audio_encoder
        torch.cuda.empty_cache()

        audio_embeddings = encoder_outputs.last_hidden_state.to('cpu')

        # To be parsed into Dataloader
        return audio_embeddings, labels 
