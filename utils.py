import torch

VOCAB_SIZE = 128256
PROMPT_SIZE = 355
EOT_TOKEN_ID = 128009

def pad_generated_seq(generated_logits, diff_in_len):
    """
    Helper function to pad the logits tensor with logits that represents the eot_id to ensure that the length of the 
    generated sequence matches the sequence length of the ground truth for loss calculation.

    Args:
        generated_logits (torch.Tensor): Input tensor with dimensions (batch_size, seq_length, vocab_size).
        diff_in_len (int): Difference in sequence length between the generated logits and the ground truth labels.
        
    Returns:
        torch.Tensor: Padded tensor with dimensions (batch_size, seq_length + diff_in_len, vocab_size).
    """
    batch_size = generated_logits.size(0)
    padding_logit = torch.zeros((batch_size, 1, VOCAB_SIZE))
    padding_logit[:, :, EOT_TOKEN_ID] = 100.0
    padding_logit = padding_logit.repeat(1, diff_in_len, 1).to(generated_logits.device)

    # Pad along sequence length dimension
    padded_logit = torch.cat((generated_logits, padding_logit), dim=1)
    return padded_logit

def padding_process(output_logits, target_ids):
    """
    1. Trims the output logits to remove the prompt logits. They are not necessary for loss calculation.
    2. Finds the difference in sequence length dimension between the output logits and the ground truth labels.
    3. Use the difference to perform padding on the output logits for smooth loss calculation. This is because
    the loss function requires an equal sequence length of the generated logits and ground truth labels.

    Args:
        output_logits (torch.Tensor): Output logits tensor with dimensions (batch_size, seq_length, vocab_size).
        target_ids (torch.Tensor): Tensor representing the tokens of the ground truth labels with dimensions (batch_size, seq_length)
        
    Returns:
        torch.Tensor: Padded tensor with dimensions (batch_size, seq_length + diff_in_len, vocab_size).
    """
    # Trim output logits
    generated_logits = output_logits[:, PROMPT_SIZE:, :]

    # Determine which to pad
    diff_in_len = target_ids.size(1) - generated_logits.size(1)
    
    if diff_in_len != 0:
        generated_logits = pad_generated_seq(generated_logits, diff_in_len)
    
    return generated_logits