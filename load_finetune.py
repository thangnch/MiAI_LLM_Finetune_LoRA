import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    HfArgumentParser, TrainingArguments, pipeline, logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
# from trl.utils import get_peft_config

# No change params
use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant = True, "float16", "nf4", False # To quantization
lora_r, lora_alpha, lora_dropout = 64, 16, 0.1
fp16, bf16 =  False, False
per_device_train_batch_size, per_device_eval_batch_size = 4, 4
gradient_accumulation_steps, gradient_checkpointing, max_grad_norm = 1, True, 0.3
learning_rate, weight_decay, optim = 2e-4, 0.001, "paged_adamw_32bit"
lr_scheduler_type, max_steps, warmup_ratio = "cosine", -1, 0.03
group_by_length, save_steps, logging_steps = True, 0, 25
max_seq_length, packing, device_map = None, False, {"": 0}

# transformed_dataset is there

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "/root/models/finetune_model.pt",
    quantization_config=bnb_config,
    device_map=device_map
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/models/tokenizer", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


# Run text generation pipeline with our next model
prompt = "Dime la definición de Psicópata."
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
