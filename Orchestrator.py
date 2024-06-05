import torch
from TranslateModel import TranslateModel
import pytorch_lightning as pl
from collections import OrderedDict


class LightningTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TranslateModel()

    def forward(self, audio_embeddings):
        output = self.model(audio_embeddings)
        return output # contains logits and sequences

    def training_step(self, batch, batch_idx):
        audio_embeddings, labels = batch

        # Get predicted tokens
        output = self.forward(audio_embeddings)
        
        # Calculate loss
        loss_value = self.calculate_loss(output.logits, labels)

        # Compile loss into a tqdm dictionary
        tqdm_dict = {'train_loss': loss_value}

        # Compile loss value with progressbar and log
        loss = OrderedDict({
            'loss': loss_value,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return loss
    

    def calculate_loss(self, logits, labels):
        """
        Cross Entropy Loss between predicted logits and output labels
        """
        # return

        pass
    
    def configure_optimizers(self):
        """
        Optimiser Configuration
        :return: list of optimizers
        """

        pass

class AudioEmbeddingsDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset Module to allow it to be converted into a DataLoader for batch processing
    """
    def __init__(self, audio_embeddings, labels):
        self.audio_embeddings = audio_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.audio_embeddings)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.labels[idx]