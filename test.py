import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from datasets import load_dataset, load_metric

model_id = "nvidia/Llama3-ChatQA-1.5-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
test_dataset= pd.read_csv('./qiaojin/updated_PubMedQA_pqa_artificial.csv')

encodings = tokenizer("\n\n".join(test_dataset["long_answer"][0:2000]))

max_length = model.config.max_length
stride = 512
seq_len = len(encodings.input_ids[0])
device = "cuda"
nlls = []
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):

    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    # input_ids = encodings.input_ids[:, begin_loc:end_loc]
    input_ids = torch.tensor(encodings.input_ids[:, begin_loc:end_loc]).to(device)
    target_ids = input_ids.clone()
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100


    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)
