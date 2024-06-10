from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TranslateModel(torch.nn.Module):
    """
    PyTorch Module for translating audio embeddings to English text using a pre-trained LLM.
    
    Attributes:
        device_type (torch.device): Device type (CUDA if available, else CPU).
        adaptor (torch.nn.Linear): Linear layer to adapt audio embeddings.
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained LLM.
        llm (AutoModelForCausalLM): Pre-trained LLM for causal language modeling.
        prefix_embeddings (torch.Tensor): Embedded prefix prompt.
        suffix_embeddings (torch.Tensor): Embedded suffix prompt.
    """
    
    def __init__(self, llm="./sea-lion-7b-instruct"):
        """
        Initializes the TranslateModel with the specified pre-trained LLM.
        
        Args:
            llm (str): Path to the pre-trained LLM.
        """
        super(TranslateModel, self).__init__()

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

        # Prevent gradient updates for the LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Embedding Prompt Format
        self.prefix_embeddings = self.embed_prompt("### USER:\nTranslate the following to English. ")
        self.suffix_embeddings = self.embed_prompt(" \n\n### RESPONSE:\n")

    def forward(self, audio_embeddings):
        """
        Forward pass of the model. Adapts audio embeddings and generates output using the LLM.
        
        Args:
            audio_embeddings (torch.Tensor): Input audio embeddings.
        
        Returns:
            output (dict): Generated output containing sequences and logits.
        """

        print(audio_embeddings.dtype)
        

        print(audio_embeddings.dtype)
        # Adapt audio embeddings
        adapted_audio_embeddings = self.adaptor(audio_embeddings)  # (batch_size, 1500, 1024)

        size = adapted_audio_embeddings.size(0)  # get batch_size of audio embeddings

        # LLM Generation Kwargs
        generation_kwargs = {
            "do_sample": False,  # set to true if temperature is not 0
            "temperature": None,
            "max_new_tokens": 30,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
        }

        # Concatenate audio embeddings with embedded prefix and suffix prompt template
        cat_embeddings = torch.cat([
            self.prefix_embeddings.repeat(size, 1, 1), 
            adapted_audio_embeddings, 
            self.suffix_embeddings.repeat(size, 1, 1)], 
            dim=1
        )
        
        # Feed into LLM
        output = self.llm.generate(
            inputs_embeds=cat_embeddings,
            **generation_kwargs,
            return_dict_in_generate=True, 
            output_logits=True
        )  # contains sequences and logits

        return output  # contains sequences and logits properties
    
    def decode(self, output):
        """
        Decodes the output sequences from the LLM to obtain the English translation.
        
        Args:
            output (dict): Generated output containing sequences.
        
        Returns:
            translated_output (list): List of translated English text.
        """
        translated_output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return translated_output
    
    def embed_prompt(self, prompt):
        """
        Embeds the given prompt using the LLM's tokenizer and embedding layer.
        
        Args:
            prompt (str): Input prompt to be embedded.
        
        Returns:
            embeddings (torch.Tensor): Embedded prompt.
        """
        tokens = self.tokenizer(prompt, return_tensors="pt")
        embeddings = self.llm.transformer.wte(tokens['input_ids'].to("cuda"))
        return embeddings