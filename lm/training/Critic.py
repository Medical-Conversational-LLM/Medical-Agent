from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from peft import LoraConfig
from easyllm.clients import huggingface
from transformers import AutoTokenizer
import pandas as pd
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer, AutoModelForCausalLM, GPT2Config


model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

df= pd.read_csv('qiaojin/critic_clean.csv', low_memory=False)
df = Dataset.from_pandas(df)


biencoder = SentenceTransformer("intfloat/e5-large-v2", device="cuda")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device=torch_device)
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
from transformers import TrainingArguments,EarlyStoppingCallback


new_model = "Llama3-ChatQA-Critic-PubMedQA"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
# num_train_epochs = 1
num_train_epochs=5.0
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
# learning_rate =  1e-5
learning_rate =  0.0011810670335804122,
weight_decay= 0.009531980022595555
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}
early_stopping = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)
   


# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    # bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5,  # Evaluate every 20 steps
    load_best_model_at_end=True,  # Load the best model found during training at the end of training
    metric_for_best_model="eval_loss",
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Instruction'])):
        text = f"###Instruction: {example['Instruction'][i]}\n ###Input: {example['Input'][i]}\n ###Task:{example['Task'][i]}   ###Response:{example['Response'][i]} \n "
        output_texts.append(text)
    return output_texts


compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=df,
    formatting_func=formatting_prompts_func,
    peft_config=peft_params,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=max_seq_length,
    packing=False,
    callbacks=[early_stopping],
)

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

config = GPT2Config.from_pretrained(model_id)
# Save the configuration to a file
config.save_pretrained("./Llama3-ChatQA-Critic-PubMedQA/")
