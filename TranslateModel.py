from datasets import Dataset, Audio
from transformers import AutoProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM
import torch
import pytorch_lightning as pl


class TranslateModel(pl.LightningModule):
    def __init__(self, audio_encoder="openai/whisper-medium", llm="sealion"):
        super().__init__()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Whisper model and processor
        self.audio_processor = AutoProcessor.from_pretrained(audio_encoder)
        self.audio_encoder = WhisperModel.from_pretrained(audio_encoder).to(self.device)

        # Freeze Audio Encoder weights
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Define the Adaptor
        self.adaptor = torch.nn.Linear(1024, 4096)  # Do we need bias?

        # Load the LLM and its tokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(llm).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(llm)

        # Freeze LLM weights
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, audio_file_path):
        # Encode Audio
        # (batch_size, 1500, 1024)

        # Adapt audio embeddings

        # Concat audio embeddings with prompt

        # Feed into LLM

        # Get translated output
        pass

    def process_and_encode_audio(self, list_audio_filepaths):
        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = self.audio_processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors='pt'
            )['input_features'][0]
            return batch

        audio_dataset = Dataset.from_dict({
            "audio": list_audio_filepaths
        })
        audio_dataset = audio_dataset.cast_column("audio", Audio())

        # Maps the audio files into Huggingface Dataset Format
        audio_dataset = audio_dataset.map(prepare_dataset)
        inputs = torch.tensor(audio_dataset['input_features']).to(self.device)

        # Ensuring No Gradient Updates during Encoding
        with torch.no_grad():
            encoder_outputs = self.audio_encoder.encoder(inputs, output_hidden_states=True)

        return encoder_outputs.last_hidden_state

    def training_step(self, batch, batch_idx):
        # Define the training step logic
        pass

    def configure_optimizers(self):
        # Define the optimizers and schedulers
        pass
