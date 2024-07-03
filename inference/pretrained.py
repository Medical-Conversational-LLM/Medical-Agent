from inference.inference import Inference
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
import torch
from utils import get_llama_formatted_prompt, list_available_devices

torch.cuda.empty_cache()

base_model_name = "nvidia/Llama3-ChatQA-1.5-8B"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

 
class Pretrained(Inference):
    model: Any
    tokenizer: Any

    def __init__(self, model_name: str) -> None:
        super().__init__()

        if model_name == base_model_name:
            self.model = base_model
        else:
            self.model = PeftModel.from_pretrained(base_model,
                                               model_name,
                                               torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2"
                                               )

        print(f"Loading: {model_name}")
        list_available_devices()

        self.model = self.model.to("cuda:0")

        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run(self, prompt: str, max_new_tokens=512, **args):
        prompt = get_llama_formatted_prompt(prompt)
        tokenized_prompt = self.tokenizer(
            self.tokenizer.bos_token + prompt, return_tensors="pt").to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = self.model.generate(
            **tokenized_prompt, max_new_tokens=max_new_tokens, eos_token_id=terminators, **args)

        return output, tokenized_prompt

    def completion(self, prompt: str, max_new_tokens=512, output_scores=False, logit=False) -> str:

        args = {'max_new_tokens': max_new_tokens}

        if output_scores == True:
            args["return_dict_in_generate"] = True,
            args["output_scores"] = True

        output, tokenized_prompt = self.run(prompt, **args)

        if output_scores:
            response = output.sequences[0][tokenized_prompt.input_ids.shape[-1]:]
            response = self.tokenizer.decode(
                response, skip_special_tokens=True).strip()
            return response, output

        response = output[0][tokenized_prompt.input_ids.shape[-1]:]
        response = self.tokenizer.decode(
            response, skip_special_tokens=True).strip()

        return response
