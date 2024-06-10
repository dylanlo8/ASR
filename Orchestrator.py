import torch
from TranslateModel import TranslateModel
import pytorch_lightning as pl
import torch.optim as optim
from transformers import AutoTokenizer
from utils import *
import torch.nn.functional as F

class LightningTranslator(pl.LightningModule):
    """
    PyTorch Lightning Module for training the TranslateModel.
    
    Attributes:
        model (TranslateModel): The translation model to be trained.
    """
    
    def __init__(self):
        super().__init__()
        self.model = TranslateModel()

    def forward(self, audio_embeddings):
        """
        Forward pass of the model. Generates output using the TranslateModel.
        
        Args:
            
        
        Returns:
            output (dict): Generated output containing logits and sequences.
        """

        output = self.model(audio_embeddings)
        return output  # contains logits and sequences

    def training_step(self, batch, batch_idx):
        """
        Training step for the model. Calculates the loss and returns it.
        
        Args:
            batch (tuple): Batch of data containing audio embeddings and labels.
            batch_idx (int): Index of the batch.
        
        Returns:
            loss (OrderedDict): OrderedDict containing the loss value, progress bar dictionary, and log dictionary.
        """

        audio_embeddings, labels = batch[0], batch[1]

        # Get predicted tokens
        output = self.forward(audio_embeddings)
        
        # Calculate loss
        loss = self.calculate_loss(output.logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Training step for the model. Calculates the loss and returns it.
        
        Args:
            batch (tuple): Batch of data containing audio embeddings and labels.
            batch_idx (int): Index of the batch.
        
        Returns:
            loss (OrderedDict): OrderedDict containing the loss value, progress bar dictionary, and log dictionary.
        """
        audio_embeddings, labels = batch[0], batch[1]

        # Get predicted tokens
        output = self.forward(audio_embeddings)
        
        # Calculate loss
        loss = self.calculate_loss(output.logits, labels)

        return loss
    
    def predict_step(self, batch, batch_idx):
        output = self.forward(batch)
        
        output_tokens = self.model.decode(output)
        return output_tokens

    def calculate_loss(self, logits, labels):
        """
        Calculates the cross-entropy loss between predicted logits and output labels.
        
        Args:
            logits (torch.Tensor): Predicted logits from the model.
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            loss_value (torch.Tensor): Calculated loss value.
        """
        
        generated_logits, labels = padding_process(logits, labels)

        # Comment out if using GPU
        # generated_logits = generated_logits.to("cpu")
        # labels = labels.to("cpu")

        loss = F.cross_entropy(
            generated_logits.reshape(-1, generated_logits.shape[-1]), labels.view(-1)
        )
        
        loss_with_grad = torch.tensor(loss.item(), requires_grad=True)

        return loss_with_grad
        
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            optimizer (torch.optim.Optimizer): Configured optimizer.
        """

        lr_default = 1.5e-3
        adam_beta1 = 0.9
        adama_beta2 = 0.999
        adam_eps = 1e-8

        # TODO: experiment with AdamW
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr_default,
            betas=(adam_beta1, adama_beta2),
            eps=adam_eps,
        )

        return optimizer


class AudioEmbeddingsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset module for audio embeddings and labels.
    
    Attributes:
        audio_embeddings (torch.Tensor): Tensor of audio embeddings.
        labels (torch.Tensor): Tensor of labels.
    """
    
    def __init__(self, audio_embeddings, transcripts):
        """
        Initializes the AudioEmbeddingsDataset with audio embeddings and audio transcript.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor of audio embeddings.
            labels (torch.Tensor): Tensor of tokenised labels.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            "./sea-lion-7b-instruct", 
            trust_remote_code=True,
            local_files_only=True
        )

        tokens = tokenizer(transcripts, return_tensors="pt", padding=True)
        tokenised_labels = tokens["input_ids"]

        self.audio_embeddings = audio_embeddings
        self.labels = tokenised_labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.audio_embeddings)

    def __getitem__(self, idx):
        """
        Retrieves the audio embedding and label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Tuple containing the audio embedding and label.
        """
        return self.audio_embeddings[idx], self.labels[idx]

if __name__ == "__main__":
    translator = LightningTranslator()