from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import *

class Adaptor(torch.nn.Module):
    """
    A Pytorch neural network module for adapting audio embeddings by applying adaptive pooling, linear projection, and layer normalization.
    The Whisper's embedded representations have the dimensions (batch, sequence_length, embedding_dimensions) which are fixed at (X, 1500, 1024), 
    where X is the batch size or size of the dataset.

    The main functionalities include:
        - Using the pooling layers to reduce the sequence length of the embedded audio dimensions to reduce memory constraints.
        - Using a Linear projection layer to project the Whisper embedding dimensions into the same dimensions as the 
        decoder-only LLM input embedding dimensions.
    
    """
    def __init__(self):
        """
        Initializes the Adaptor with adaptive average pooling layers, a linear projection layer, and a layer normalization layer.
        """
        super(Adaptor, self).__init__()
        self.pool1 = torch.nn.AdaptiveAvgPool1d(output_size=750)
        self.pool2 = torch.nn.AdaptiveAvgPool1d(output_size=325)
        self.linear = torch.nn.Linear(1024, 4096)
        self.layer_norm = torch.nn.LayerNorm(4096)

    def forward(self, x):
        """
        Forward pass of the Adaptor.
        
        Args:
            x (torch.Tensor): Input tensor with dimensions (batch_size, seq_length, embedding_dim).
        
        Returns:
            torch.Tensor: Adapted tensor with dimensions (batch_size, new_sequence_length, new_embedding_dim).
        """
        # Apply mean pooling along the sequence_length dimension, rather than the embedding_dim
        x = self.pool1(x.permute(0, 2, 1))  # Permute input to pool along seq_length dimension
        x = self.pool2(x)
         
        # Apply linear projection along embedding dim
        x = x.permute(0, 2, 1)  # Permute back to the original format (batch_size, seq_length, embedding_dim)
        x = self.linear(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        return x

class TranslateModel(torch.nn.Module):
    """
    Model for translating audio embeddings into text using a pre-trained language model.

    This class provides methods to load a pre-trained Large Language Model, adapt audio embeddings, and 
    pass the adapted audio embeddings as inputs into the LLM for text translations.
    
    Attributes:
        device_type (torch.device): Device to run the model on (CUDA or CPU).
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained SeaLion Large Language Model.
        llm (AutoModelForCausalLM): Pre-trained language model for causal language modeling.
        prefix_embeddings (torch.Tensor): Embeddings for the prefix prompt template.
        suffix_embeddings (torch.Tensor): Embeddings for the suffix prompt template.
        adaptor (Adaptor): Neural network module for adapting audio embeddings.
        generation_kwargs (dict): Arguments for text generation.
    """
    
    def __init__(self, llm="./sea-lion-7b-instruct"):
        """
        Initializes the TranslateModel with the specified pre-trained language model.
        
        Args:
            llm (str): Path to the pre-trained language model.
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

        # Freeze the LLM parameters to prevent training of the LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Embedding the SeaLion LLM Prompt Template
        self.prefix_embeddings = self.embed_prompt("### USER:\nTranslate the following to English. ").to(self.device_type)
        self.suffix_embeddings = self.embed_prompt("\n\n### RESPONSE:\n").to(self.device_type)

        # Initialise the adaptor
        self.adaptor = Adaptor()

        for param in self.adaptor.parameters():
            param.requires_grad = True

        # Defining generation arguments for inference
        self.generation_kwargs = {
            "do_sample": False,  # set to true if temperature is not 0
            "temperature": None,
            "max_new_tokens": 256,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.eos_token_id
        }

    def forward(self, audio_embeddings, transcripts, eos_token_id=1):
        """
        Forward pass for translating audio embeddings to text.
        
        The main functionalities include:
            1. Tokenising the transcript into tokenised labels.
            2. Project the Whisper audio embeddings through the Adaptor module and concatenating
            it with the LLM prompt template to prepare it as inputs into the LLM.
            3. Iteratively predict the next token by using the previous generated token as inputs.
            4. Returns logits as output from the forward pass through the LLM.
        
        Args:
            audio_embeddings (torch.Tensor): Audio embedding tensor with dimensions (batch_size, sequence_length = 1500, embedding_dim = 1024).
            transcripts (list of str): List of transcripts corresponding to the audio inputs.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 1.
        
        Returns:
            logits (torch.tensor): Logits tensor with dimensions (batch_size, sequence_length, vocab_size = 256000)
            attention_mask (torch.tensor): Attention mask pytorch tensor with dimensions (batch_size, sequence_length)
        """

        # Tokenising the transcriptions
        labels = self.tokenizer(transcripts, return_tensors='pt')['input_ids']
        max_tokens_to_generate = labels.size(1)

        # Forward pass through the Adaptor module
        adapted_audio_embeddings = self.adaptor(audio_embeddings)  # (batch_size, adapted_seq, 1024)
        adapted_audio_embeddings = adapted_audio_embeddings.to(self.device_type)
        batch_size = adapted_audio_embeddings.size(0)

        # Concatenate audio embeddings with embedded prefix and suffix prompt template
        cat_embeddings = torch.cat([
            self.prefix_embeddings.repeat(batch_size, 1, 1), 
            adapted_audio_embeddings, 
            self.suffix_embeddings.repeat(batch_size, 1, 1)], 
            dim=1
        )

        inputs_embeds = cat_embeddings

        # Initialise a tensor to mark whether a sample in a batch has generated the eos_token
        eos_mask = torch.zeros(batch_size, dtype=torch.bool)

        # Pairs the eos_mask with a binary masked tensor where predictions at positions with 1 will be
        # considered during the loss calculation and elements with 0 will be ignored at the loss calculation
        attention_mask = torch.ones((batch_size, max_tokens_to_generate)).to("cuda")
        
        # Iteratively generate tokens 
        for i in range(max_tokens_to_generate):
            # Forward pass
            res = self.llm(inputs_embeds=inputs_embeds)

            # Extract the token id with the highest predicted probability
            sampled_token = torch.argmax(res.logits[:, -1, :].softmax(dim=-1), -1).view(-1, 1)

            # Convert the sampled token into embedding
            sampled_embedding = self.llm.transformer.wte(sampled_token)

            # Concat sampled token embedding with the input embeddings for the next token prediction
            inputs_embeds = torch.cat((inputs_embeds, sampled_embedding), dim=1)  # batch_size, seq_length, 4096

            # Update eos_mask to mark sequences that have generated eos_token_id
            for batch_idx in range(batch_size):
                # If sampled token == endoftext token
                if not eos_mask[batch_idx] and sampled_token[batch_idx, 0].item() == eos_token_id:
                    eos_mask[batch_idx] = True

                    # all tokens from here set to 0 in attention mask
                    if i < max_tokens_to_generate:  # prevent out of bounds error
                        attention_mask[batch_idx, i + 1:] = 0  # ele at i represent end_of_text token
                    
            # Check if all sequences have generated eos_token_id
            if eos_mask.all():
                attention_mask = attention_mask[:, :i + 1]
                break

        return res.logits, attention_mask 
    
    def predict(self, audio_embeddings):
        """
        Generates text translations for a given audio embedding. This function adapts the audio embedding with the Adaptor module
        and concatenates the adapted audio embedding with the LLM prompt template. This concatenated embedding is passed
        into the generate function of the LLM for token generation.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor containing audio embeddings.
        
        Returns:
            torch.Tensor: Generated tokenised output of dimensions (batch_size, sequence_length). 
            This tensor needs to be decoded by the tokeniser to convert it into string output.
        """

        # Adapt audio embeddings
        adapted_audio_embeddings = self.adaptor(audio_embeddings.to(self.device_type))  # (batch_size, adapted_seq, 1024)

        batch_size = adapted_audio_embeddings.size(0)  # get batch_size of audio embeddings

        # Concatenate audio embeddings with embedded prefix and suffix prompt template
        cat_embeddings = torch.cat([
            self.prefix_embeddings.repeat(batch_size, 1, 1), 
            adapted_audio_embeddings, 
            self.suffix_embeddings.repeat(batch_size, 1, 1)], 
            dim=1
        )

        output = self.llm.generate(
            inputs_embeds=cat_embeddings,
            **self.generation_kwargs)
        
        return output
    
    def decode(self, logits, attention_mask):
        """
        Applies a softmax function on the logits and return the argmax of the probabilities of the predicted tokens.
        Returns a pytorch tensor of the predicted tokens 
        
        Args:
            logits (torch.Tensor): Logits tensor with dimensions (batch_size, sequence_length, vocab_size=256000).
            attention_mask (torch.Tensor): Attention mask tensor.
        
        Returns:
            torch.Tensor: Sampled token IDs with dimensions (batch_size, sequence_length).
        """
        logits = post_process_logits(logits, attention_mask)
        
        # Softmax along Vocabulary dimension 
        probs = logits.softmax(dim=-1)

        # Extract the token IDs which has the highest probability
        sampled_tokens = torch.argmax(probs, -1)
        
        return sampled_tokens
    
    def embed_prompt(self, prompt):
        """
        Embeds the given prompt into token embeddings.
        
        Args:
            prompt (str): Text prompt to be embedded.
        
        Returns:
            torch.Tensor: Embeddings for the prompt with dimensions (sequence_length, embedding_dim=4096).
        """
        tokens = self.tokenizer(prompt, return_tensors="pt")
        embeddings = self.llm.transformer.wte(tokens['input_ids'].to("cuda"))
        return embeddings
