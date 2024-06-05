from transformers import WhisperModel
import torch

class Whisper:
    def __init__(self, audio_encoder="./whisper-medium"):  
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Whisper")
        self.audio_encoder = WhisperModel.from_pretrained(audio_encoder, local_files_only=True).to(self.device_type)
        print("Whisper Loaded and ready to embed audio inputs")
        
    def embed_audio(self, audio_inputs, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.audio_encoder.encoder(
                audio_inputs, 
                output_hidden_states=True,
                attention_mask = attention_mask
            )

        del self.audio_encoder
        return encoder_outputs.last_hidden_state.to('cpu')
    
