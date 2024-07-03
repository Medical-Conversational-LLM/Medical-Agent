import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.pretrained import Pretrained
import os

torch.cuda.empty_cache()
model = Pretrained("nvidia/Llama3-ChatQA-1.5-8B")
device='cuda:1'
def preplixity():

    tokenizer = model.tokenizer
    # model_id = "NousResearch/Llama-2-13b-hf"
    # model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # # Check if tokenizer has padding token
    if tokenizer.pad_token is None:
        # If tokenizer does not have a padding token, set it to eos_token
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})




    test_dataset = pd.read_csv('storage/datasets/updated_PubMedQA_pqa_artificial.csv')

    # Tokenize all the text in your dataset with a specified max_length
    encodings = tokenizer("\n\n".join(test_dataset["long_answer"][0:2000]), return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Extract necessary variables
    max_length = model.peft_config.n_positions if hasattr(model.peft_config, 'n_positions') else model.peft_config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)  # Get the length of the input sequence

    nlls = []
    prev_end_loc = 0

    # Loop through the tokenized sequences
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc
        
        # Slice the input_ids and target_ids
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask out the future tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)  # Ensure tensors are on the same device
            neg_log_likelihood = outputs.loss  # Retrieve the negative log likelihood loss

        nlls.append(neg_log_likelihood.item())  # Append the loss value

    # Compute perplexity
    average_nll = sum(nlls) / len(nlls)  # Calculate the average negative log likelihood
    ppl = torch.exp(torch.tensor(average_nll))  # Compute perplexity as exponential of average NLL

    print(f"Perplexity: {ppl.item()}")

preplixity()