from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import *

class Adaptor(torch.nn.Module):
    """
    Adaptor module to adapt audio embeddings.
    """
    def __init__(self):
        super(Adaptor, self).__init__()
        self.pool1 = torch.nn.AdaptiveAvgPool1d(output_size=750)
        self.pool2 = torch.nn.AdaptiveAvgPool1d(output_size=325)
        #self.pool3 = torch.nn.AdaptiveAvgPool1d(output_size=10)
        self.linear = torch.nn.Linear(1024, 4096)
        self.layer_norm = torch.nn.LayerNorm(4096)  # Adding LayerNorm

    def forward(self, x):
        # Apply adaptive pooling along sequence length
        x = self.pool1(x.permute(0, 2, 1))  # Permute input to pool along seq
        x = self.pool2(x)
        #x = self.pool3(x)
         
        # Apply linear projection along embedding dim
        x = x.permute(0, 2, 1)  # Permute back to the original format (batch, seq, embed_dim)
        x = self.linear(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        return x

class TranslateModel(torch.nn.Module):
    """
    PyTorch Module for translating audio embeddings to English text using a pre-trained LLM.
    
    Attributes:
        device_type (torch.device): Device type (CUDA if available, else CPU).
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained LLM.
        llm (AutoModelForCausalLM): Pre-trained LLM for causal language modeling.
        prefix_embeddings (torch.Tensor): Embedded prefix prompt.
        suffix_embeddings (torch.Tensor): Embedded suffix prompt.
        adaptor (Adaptor): Adaptor for adapting audio embeddings.
    """
    
    def __init__(self, llm="./sea-lion-7b-instruct"):
        """
        Initializes the TranslateModel with the specified pre-trained LLM.
        
        Args:
            llm (str): Path to the pre-trained LLM.
        """
        super(TranslateModel, self).__init__()

        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            device_map="cuda",
            local_files_only=True
        )

        # Prevent gradient updates for the LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Embedding Prompt Format
        self.prefix_embeddings = self.embed_prompt("### USER:\nTranslate the following to English. ")
        self.suffix_embeddings = self.embed_prompt("\n\n### RESPONSE:\n")

        # Initialize the adaptor
        self.adaptor = Adaptor()

        for param in self.adaptor.parameters():
            param.requires_grad = True

    def forward(self, audio_embeddings, max_iterations = 140, eos_token_id = 1):
        """
        Forward pass of the model. Adapts audio embeddings and generates output using the LLM.
        
        Args:
            audio_embeddings (torch.Tensor): Input audio embeddings.
        
        Returns:
            output (dict): Generated output containing sequences and logits.
        """

        # Adapt audio embeddings
        adapted_audio_embeddings = self.adaptor(audio_embeddings)  # (batch_size, adapted_seq, 1024)

        batch_size = adapted_audio_embeddings.size(0)  # get batch_size of audio embeddings

        # Concatenate audio embeddings with embedded prefix and suffix prompt template
        cat_embeddings = torch.cat([
            self.prefix_embeddings.repeat(batch_size, 1, 1), 
            adapted_audio_embeddings, 
            self.suffix_embeddings.repeat(batch_size, 1, 1)], 
            dim=1
        )

        inputs_embeds = cat_embeddings
        eos_mask = torch.zeros(batch_size, dtype=torch.bool)
        attention_mask = torch.ones((batch_size, max_iterations)).to("cuda")

        for i in range(max_iterations):
            torch.cuda.empty_cache()
            # Forward pass
            res = self.llm(inputs_embeds=inputs_embeds)
            
            torch.cuda.empty_cache()

            # Sample using multinomial on the logits of the last tokens in each sequence
            sampled_token = torch.argmax(res.logits[:, -1, :].softmax(dim=-1), -1).view(-1, 1)

            # convert sampled token into embedding
            sampled_embedding = self.llm.transformer.wte(sampled_token)

            # Concatenate the sampled tokens to the outputs for the next iteration
            inputs_embeds = torch.cat((inputs_embeds, sampled_embedding), dim = 1) # batch, seqlength, 4096

            # Update eos_mask to mark sequences that have generated eos_token_id
            for batch_idx in range(batch_size):
                # If sampled token == endoftext
                if not eos_mask[batch_idx] and sampled_token[batch_idx, 0].item() == eos_token_id:
                    eos_mask[batch_idx] = True

                    # all tokens from here set to 0
                    if i < max_iterations: # prevent out of bounds error
                        attention_mask[batch_idx, i+1:] = 0 # ele at i represent end_of_text token
                    
            # Check if all sequences have generated eos_token_id
            if eos_mask.all():
                attention_mask = attention_mask[:,:i+1]
                break

        return res.logits, attention_mask  # contains sequences and logits properties
    
    def decode(self, logits, attention_mask):
        logits = post_process_logits(logits, attention_mask)
        
        # Softmax along vocab 
        probs = logits.softmax(dim=-1)


        # Sample from the probability distribution
        sampled_tokens = torch.argmax(probs, -1)
        
        return sampled_tokens
    
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
