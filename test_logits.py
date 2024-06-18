from inference.inference import Inference
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
import torch
from utils import get_llama_formatted_prompt

base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Llama3-ChatQA-1.5-8B")


model_name = "HlaH/Llama3-ChatQA-Generator-PubMedQA"

model = PeftModel.from_pretrained(base_model,
                                  model_name,
                                  torch_dtype=torch.bfloat16,
                                  attn_implementation="flash_attention_2"
                                  ).to("cuda")

model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = [
    {
        "role": "user",
        "content": "what is cancer"
    }
]

prompt = get_llama_formatted_prompt(prompt)
tokenized_prompt = tokenizer(
    tokenizer.bos_token + prompt, return_tensors="pt").to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model(**tokenized_prompt, return_dict=True)


print(outputs.logits)
