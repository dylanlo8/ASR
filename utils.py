import torch

# Declare variables as ENV 
vocab_size = 256000
prompt_size = 342 # 18 # 342

def post_process_logits(output_logits, output_mask):

    # Trim output logits to get the output without prompts
    trimmed_logits = output_logits[:, prompt_size:, :]

    # Create logit that represents eot token
    eot_logit = torch.zeros((1, 1, vocab_size)).to("cuda")
    eot_logit[:, :, 1] = 1.0
    attention_mask_bool = output_mask == 0

    expanded_mask = attention_mask_bool.unsqueeze(-1).expand(-1, -1, vocab_size)
    
    logits = torch.where(expanded_mask, eot_logit, trimmed_logits)

    return logits

def compare_generated_and_actual(prob_logits_transposed, target_ids):    
    # Get max length of sequence generated
    max_len_generated_int = prob_logits_transposed.size(1)
    # Get max length of original captions
    max_len_original_int = target_ids.size(1)
    # Calculate difference between the lengths
    diff_in_len = max_len_generated_int - max_len_original_int
    return diff_in_len

def pad_actual(target_ids, diff_in_len):
    # Pad on right side of last dimension
    padding_template = (0, diff_in_len)
    # Pad actual tokens with <end_of_text>
    target_ids = torch.nn.functional.pad(target_ids, padding_template, "constant", 1)
    # Return modified actual sequence
    return target_ids

def pad_generated_seq(generated_logits, diff_in_len):
    # Create padding tensor that represents <end_of_text> with token id 1
    batch_size = generated_logits.size(0) # index 0 of generated logits gives the batch size 
    padding_logit = torch.zeros((batch_size, 1, vocab_size))
    padding_logit[:, :, 1] = 1.0
    padding_logit = padding_logit.repeat(1, diff_in_len, 1).to(generated_logits.device)
    # Pad generated sequence along dimension 1 (length of sequence)
    padded_logit = torch.cat((generated_logits, padding_logit), dim=1)
    # Return padded logit
    return padded_logit

def padding_process(output_logits, output_mask, target_ids):
    # Transpose output logits to suitable format
    generated_logits = post_process_logits(output_logits, output_mask)

    # Determine which to pad
    diff_in_len = compare_generated_and_actual(generated_logits, target_ids)

    # Case 1: Generated sequence is longer than actual
    if (diff_in_len > 0):
        target_ids = pad_actual(target_ids, abs(diff_in_len))
    # Case 2: Generated sequence is shorter than actual
    elif (diff_in_len < 0):
        generated_logits = pad_generated_seq(generated_logits, abs(diff_in_len))

    return generated_logits, target_ids