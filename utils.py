import torch

# Declare variables as ENV 
vocab_size = 256000
batch_size = 2

def process_output_logits(output_logits):
    # Convert tuple of tensors to a single tensor with 3D dimensions
    stacked_logits = torch.stack(output_logits)
    # Remove the last token, since it represents </end_of_text>
    stacked_logits = stacked_logits[:-1][:][:]
    # Convert logits to probabilities
    prob_logits = stacked_logits.softmax(dim=-1)
    # Dimensions of prob_logits: [max_length_of_generated_seq][batch_size][vocab_size]
    prob_logits_transposed = prob_logits.transpose(0, 1)
    # Dimensions of transposed: [batch_size][max_length_of_generated_seq][vocab_size]
    return prob_logits_transposed

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
    padding_logit = torch.zeros((batch_size, 1, vocab_size))
    padding_logit[0,0,1] = 1.0
    padding_logit = padding_logit.repeat(1, diff_in_len, 1)
    # Pad generated sequence along dimension 1 (length of sequence)
    padded_logit = torch.cat((generated_logits, padding_logit), dim=1)
    # Return padded logit
    return padded_logit

def padding_process(output_logits, target_ids):
    # Transpose output logits to suitable format
    generated_logits = process_output_logits(output_logits)
    # Determine which to pad
    diff_in_len = compare_generated_and_actual(generated_logits, target_ids)

    # Case 1: Generated sequence is longer than actual
    if (diff_in_len > 0):
        target_ids = pad_actual(target_ids, diff_in_len)
    # Case 2: Generated sequence is shorter than actual
    elif (diff_in_len < 0):
        generated_logits = pad_generated_seq(generated_logits, diff_in_len)
    # Case 3: Same maximum length
    else:
        print("Max length of generated = Max length of actual in the batch")

    return generated_logits, target_ids