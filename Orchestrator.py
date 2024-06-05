import torch
from TranslateModel import TranslateModel
import pytorch_lightning as pl
from collections import OrderedDict

class LightningTransformer(pl.LightningModule):
    """
    PyTorch Lightning Module for training the TranslateModel.
    
    Attributes:
        model (TranslateModel): The translation model to be trained.
    """
    
    def __init__(self):
        """
        Initializes the LightningTransformer with the TranslateModel.
        """
        super().__init__()
        self.model = TranslateModel()

    def forward(self, audio_embeddings):
        """
        Forward pass of the model. Generates output using the TranslateModel.
        
        Args:
            audio_embeddings (torch.Tensor): Input audio embeddings.
        
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
        audio_embeddings, labels = batch

        # Get predicted tokens
        output = self.forward(audio_embeddings)
        
        # Calculate loss
        loss_value = self.calculate_loss(output.logits, labels)

        # Compile loss into a tqdm dictionary
        tqdm_dict = {'train_loss': loss_value}

        # Compile loss value with progress bar and log
        loss = OrderedDict({
            'loss': loss_value,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return loss

    def calculate_loss(self, logits, labels):
        """
        Calculates the cross-entropy loss between predicted logits and output labels.
        
        Args:
            logits (torch.Tensor): Predicted logits from the model.
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            loss_value (torch.Tensor): Calculated loss value.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_value = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss_value
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            optimizer (torch.optim.Optimizer): Configured optimizer.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

class AudioEmbeddingsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset module for audio embeddings and labels.
    
    Attributes:
        audio_embeddings (torch.Tensor): Tensor of audio embeddings.
        labels (torch.Tensor): Tensor of labels.
    """
    
    def __init__(self, audio_embeddings, labels):
        """
        Initializes the AudioEmbeddingsDataset with audio embeddings and labels.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor of audio embeddings.
            labels (torch.Tensor): Tensor of labels.
        """
        self.audio_embeddings = audio_embeddings
        self.labels = labels

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
