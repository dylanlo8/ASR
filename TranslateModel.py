from datasets import Dataset, Audio
from transformers import AutoProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import pytorch_lightning as pl


class TranslateModel(pl.LightningModule):
    def __init__(self, audio_encoder="openai/whisper-medium", llm="aisingapore/sealion7b-instruct-nc"):
        super().__init__()

        # Device
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Whisper model and processor
        print("Loading Audio Encoder")
        self.audio_processor = AutoProcessor.from_pretrained(audio_encoder)
        self.audio_encoder = WhisperModel.from_pretrained(audio_encoder).to(self.device_type)

        # Define the Adaptor
        self.adaptor = torch.nn.Linear(1024, 4096)  # Do we need bias?

        # Load the LLM and its tokenizer
        print("Loading LLM")

        self.generation_kwargs = {
            "do_sample": False,  # set to true if temperature is not 0
            "temperature": None,
            "max_new_tokens": 256,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm, 
            trust_remote_code=True
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm, 
            trust_remote_code=True,
            device_map="auto"
        )

        self.prefix_embeddings = self.embed_prompt_tokens("### USER:\nTranslate the following to English. ")
        self.suffix_embeddings = self.embed_prompt_tokens(" \n\n### RESPONSE:\n")


    def forward(self, list_audio_filepaths):
        # Encode Audio
        # (batch_size, 1500, 1024)
        audio_embeddings = self.process_and_encode_audio(list_audio_filepaths)

        # Adapt audio embeddings
        adapted_audio_embeddings = self.adaptor(audio_embeddings)

        # Concat audio embeddings with prompt
        torch.cat([self.prefix_embeddings, adapted_audio_embeddings, self.suffix_embeddings])

        input_embeddings = torch.cat([self.prefix_embeddings.unsqueeze(0), adapted_audio_embeddings, self.suffix_embeddings.unsqueeze(0)], dim=1)

        # Feed into LLM
        tokenised_output = self.llm.generate(
            input_embeds = input_embeddings,
            **self.generation_kwargs
        )

        # Get translated output
        translated_output = self.tokenizer.decode(
            tokenised_output[0], 
            skip_special_tokens= True
        )

        return translated_output

    def process_and_encode_audio(self, list_audio_filepaths):
        print("Loading Dataset")
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
        audio_dataset = audio_dataset.map(prepare_dataset, num_proc=1)
        inputs = torch.tensor(audio_dataset['input_features']).to(self.device_type)

        # Ensuring No Gradient Updates during Encoding
        with torch.no_grad():
            encoder_outputs = self.audio_encoder.encoder(inputs, output_hidden_states=True)

        return encoder_outputs.last_hidden_state
    
    def embed_prompt_tokens(self, string):
        tokens = self.tokenizer(string, return_tensors="pt")
        token_embeddings = self.llm.transformer.wte(tokens['input_ids'])
        return token_embeddings
    

    def training_step(self, batch, batch_idx):
        # Define the training step logic
        pass

    def configure_optimizers(self):
        # Define the optimizers and schedulers
        pass


def main():
    # Initialize TranslateModel
    translate_model = TranslateModel()

    # List of audio file paths
    list_audio_filepaths = ["data/sub/De95Osq7p1c_trimmed_segment_1.wav", "data/sub/De95Osq7p1c_trimmed_segment_2.wav"]

    # Forward pass through the model
    translated_output = translate_model(list_audio_filepaths)

    # Print translated output
    print("Translated Output:", translated_output)

if __name__ == "__main__":
    main()