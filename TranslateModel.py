from datasets import Dataset, Audio
from transformers import AutoProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import pytorch_lightning as pl


class TranslateModel(torch.nn.Module):
    def __init__(self, llm="./sea-lion-7b-instruct"):
        super(TranslateModel, self).__init__()

        # Device
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the Adaptor
        self.adaptor = torch.nn.Linear(1024, 4096)  # Do we need bias?

        # Load the LLM and its tokenizer
        print("Loading LLM")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm, 
            trust_remote_code=True,
            local_files_only=True
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True
        )

        # Embedding Prompt Format
        self.prefix_embeddings = self.embed_prompt("### USER:\nTranslate the following to English. ")
        self.suffix_embeddings = self.embed_prompt(" \n\n### RESPONSE:\n")

    def forward(self, batch):
        torch.cuda.empty_cache()
        # Adapt audio embeddings
        adapted_batch = self.adaptor(batch) # (batch_size, 1500, 1024)

        size = adapted_batch.size(0)

        # LLM Generation Kwargs
        generation_kwargs = {
            "do_sample": False,  # set to true if temperature is not 0
            "temperature": None,
            "max_new_tokens": 30,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
        }

        input_embeddings = torch.cat([
            self.prefix_embeddings.repeat(size, 1, 1), 
            adapted_batch, 
            self.suffix_embeddings.repeat(size, 1, 1)], 
            dim=1
        )
        
        input_embeddings = input_embeddings.to("cuda")

        # Feed into LLM
        output = self.llm.generate(
            inputs_embeds = input_embeddings,
            **generation_kwargs,
            return_dict_in_generate=True, 
            output_logits=True
        ) # contains sequences and logits

        return output
        
    def decode(self, output):
        # To obtain English translation, if logits is not required
        translated_output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return translated_output
    
    def embed_prompt(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt")
        embeddings = self.llm.transformer.wte(tokens['input_ids'])
        return embeddings.to("cpu")