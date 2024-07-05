import torch
from TranslateModel import TranslateModel
import pytorch_lightning as pl
import torch.optim as optim
from transformers import AutoTokenizer
from utils import *
import torch.nn.functional as F
import copy

torch.set_grad_enabled(True)

class LightningTranslator(pl.LightningModule):
    """
    PyTorch Lightning Module for training the TranslateModel.
    """
    
    def __init__(self):
        super().__init__()
        self.model = TranslateModel()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./sea-lion-7b-instruct", 
            trust_remote_code=True,
            local_files_only=True
        )

        # self.automatic_optimization = False
        
    def forward(self, audio_embeddings, transcripts):
        logits, mask = self.model(audio_embeddings, transcripts)
        return logits, mask

    def training_step(self, batch, batch_idx):
        audio_embeddings, transcripts = batch[0], batch[1]

        # Tokenise transcripts
        tokens = self.tokenizer(transcripts, return_tensors="pt", padding=True)
        tokenised_labels = tokens["input_ids"].to("cuda")

        # Get predicted tokens
        output_logits, attention_mask = self(audio_embeddings, transcripts)
        
        print("\n")
        print(self.tokenizer.batch_decode(self.model.decode(output_logits, attention_mask)))
        print("\n")
        print(transcripts)
        print("\n")

        # opt = self.optimizers
        # opt.zero_grad()

        # Calculate loss
        loss = self.calculate_loss(output_logits, attention_mask, tokenised_labels)

        # self.manual_backward(loss)
        # opt.step()
        # sch = self.lr_schedulers()
        # sch.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def check_adaptor_gradients(self):
        for name, param in self.model.adaptor.named_parameters():
            if param.grad is not None:
                print(f"Adaptor Parameter {name}:")
                print(f" - Gradient Mean: {param.grad.mean()}")
                print(f" - Gradient Std: {param.grad.std()}")
            else:
                print(f"Adaptor Parameter {name}: No gradient")
    
    def predict_step(self, batch, batch_idx):
        audio_embeddings, transcripts = batch[0], batch[1]
        output = self.model.predict(audio_embeddings)
        
        print("\n")
        print(self.tokenizer.batch_decode(output, skip_special_tokens=False))
        print("\n")
        print(transcripts)
        print("\n")

        return output
        
    def calculate_loss(self, logits, mask, labels):
        generated_logits, labels = padding_process(logits, mask, labels)

        # Ignore padding tokens
        loss = torch.nn.CrossEntropyLoss(ignore_index = 3)(
            generated_logits.permute(0, 2, 1), labels
        )
        
        return loss
        
    def configure_optimizers(self):
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

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

        return [optimizer], [{"scheduler" : lr_scheduler, "interval" : "epoch"}]


class AudioEmbeddingsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset module for audio embeddings and labels.
    
    Attributes:
        audio_embeddings (torch.Tensor): Tensor of audio embeddings.
        transcript (torch.Tensor): Tensor of transcripts.
    """
    
    def __init__(self, audio_embeddings, transcripts):
        """
        Initializes the AudioEmbeddingsDataset with audio embeddings and audio transcript.
        
        Args:
            audio_embeddings (torch.Tensor): Tensor of audio embeddings.
            labels (torch.Tensor): Tensor of tokenised labels.
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
        Retrieves the audio embedding and label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Tuple containing the audio embedding and label.
        """
        return self.audio_embeddings[idx], self.transcripts[idx]

if __name__ == "__main__":
    translator = LightningTranslator()