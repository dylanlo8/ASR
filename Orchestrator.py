import torch
from TranslateModel import TranslateModel
import pytorch_lightning as pl
import torch.optim as optim
from transformers import AutoTokenizer
from utils import *
import torch.nn.functional as F
import config

torch.set_grad_enabled(True)

class LightningTranslator(pl.LightningModule):
    """
    PyTorch Lightning module for training and inference of the TranslateModel.

    This class provides methods for training, validation, prediction, and configuring optimizers for the TranslateModel. It handles the 
    forward pass, loss calculation, and gradient logging.

    Attributes:
        model (TranslateModel): Instance of the TranslateModel class.
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained language model.
    """
    
    def __init__(self):
        super().__init__()
        self.model = TranslateModel()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM['LLM_NAME'], 
            trust_remote_code=True,
            local_files_only=True
        )
        
    def forward(self, audio_embeddings, tokenised_labels):
        """
        Forward pass for generating logits and attention mask from audio embeddings and transcripts.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor containing audio embeddings.
            tokenised_labels (torch.Tensor): Tensor containing the tokens of the ground truth labels.
        
        Returns:
            torch.Tensor: Output logits tensor with dimensions (batch_size, seq_length, vocab_size)
        """
        logits = self.model(audio_embeddings, tokenised_labels)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step for processing a batch of audio embeddings and transcripts.

        Args:
            batch (tuple): Batch of data containing audio embeddings and transcripts tuples.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Calculated loss for the batch.
        """
        audio_embeddings, transcripts = batch[0], batch[1]

        # Tokenize transcripts
        tokens = self.tokenizer(transcripts, return_tensors="pt")
        tokenised_labels = tokens["input_ids"][:, 1:].to("cuda") # Indexing to skip <|begin_of_text|> token

        # Get predicted tokens
        output_logits = self(audio_embeddings, tokenised_labels)
        
        # Calculate cross entropy loss
        loss = self.calculate_loss(output_logits, tokenised_labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Check whether gradients are being backpropogated if necessary
        # self.check_adaptor_gradients()
        return loss

    def check_adaptor_gradients(self):
        """
        Test function to check whether gradients are being backpropogated to the Adaptor module of the model.
        """
        for name, param in self.model.adaptor.named_parameters():
            if param.grad is not None:
                print(f"Adaptor Parameter {name}:")
                print(f" - Gradient Mean: {param.grad.mean()}")
                print(f" - Gradient Std: {param.grad.std()}")
            else:
                print(f"Adaptor Parameter {name}: No gradient")
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step for processing a batch of audio embeddings and transcripts.

        Args:
            batch (tuple): Batch of data containing audio embeddings and transcripts.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Generated output from the model.
        """
        audio_embeddings, transcripts = batch[0], batch[1]
        output = self.model.predict(audio_embeddings)

        print("\n")
        print(self.tokenizer.batch_decode(output, skip_special_tokens=False))
        print("\n")
        print(transcripts)
        print("\n")

        return output
        
    def calculate_loss(self, logits, tokenised_labels):
        """
        Calculates the cross entropy loss between predicted logits and tokenized labels.

        Args:
            logits (torch.Tensor): Output logits from the forward function.
            tokenised_labels (torch.Tensor): Tensor containing the tokens of the ground truth labels.
        
        Returns:
            torch.Tensor: Calculated Cross Entropy Loss. Pytorch Lightning handles backpropogation under the hood.
        """

        # Pad logits if needed
        generated_logits = padding_process(logits, tokenised_labels)

        # Require permuting logits into (batch_size, vocab, sequence_length)
        loss = torch.nn.CrossEntropyLoss(ignore_index = 128002)(
            generated_logits.permute(0, 2, 1), tokenised_labels
        )
        
        return loss
        
    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for training.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        lr_default = 1.5e-3
        adam_beta1 = 0.9
        adama_beta2 = 0.999
        adam_eps = 1e-8

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr_default,
            betas=(adam_beta1, adama_beta2),
            eps=adam_eps,
        )

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

class AudioEmbeddingsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset module for audio embeddings and labels.
    
    Attributes:
        audio_embeddings (torch.Tensor): Tensor of audio embeddings.
        transcript (torch.Tensor): Tensor of transcripts.
    """
    
    def __init__(self, audio_embeddings, transcripts):
        """
        Initializes the AudioEmbeddingsDataset with audio embeddings and audio transcripts.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor of audio embeddings.
            transcripts (torch.Tensor): Tensor of transcripts.
        """
        self.audio_embeddings = audio_embeddings
        self.transcripts = transcripts

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.audio_embeddings)

    def __getitem__(self, idx):
        """
        Retrieves the audio embedding and transcript at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Tuple containing the audio embedding and transcript.
        """
        return self.audio_embeddings[idx], self.transcripts[idx]

if __name__ == "__main__":
    translator = LightningTranslator()
