"""
RoadWatch LLM Fine-Tuning Script
This script provides the boilerplate to fine-tune HuggingFaceTB/SmolLM2-1.7B-Instruct 
using PEFT (LoRA) and TRL.

Usage:
1. Run this in Google Colab (with a T4 or A100 GPU) or a local machine with an Nvidia GPU.
2. Install requirements: pip install transformers peft trl datasets torch accelerate
3. Login to Hugging Face: huggingface-cli login
4. Run: python train_llm.py
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
NEW_MODEL = "RoadWatch-SmolLM2-1.7B-Instruct-v1"
HUGGINGFACE_USERNAME = "YOUR_HF_USERNAME" # Update this!

# Prepare a tiny synthetic dataset based on our Knowledge Base
# (In a real scenario, you would load this from a large JSON/CSV file)
custom_data = {
    "text": [
        "User: How do I report a pothole on a National Highway?\nAssistant: You can report it to the NHAI Regional Office. Their helpline is 1033.",
        "User: What is the PMGSY 5-year rule?\nAssistant: Under PMGSY, contractors have a mandatory 5-year maintenance contract. The funds are held in an escrow account.",
        "User: Who maintains Village roads?\nAssistant: Village roads are maintained by the Gram Panchayat or DRDA. The helpline is 104.",
    ]
}

def main():
    print("[*] Loading dataset...")
    dataset = Dataset.from_dict(custom_data)

    print(f"[*] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[*] Configuring LoRA (PEFT)...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    print("[*] Loading base model...")
    # For a 1.7B model, you can load it in bfloat16 or 4-bit on a free Colab T4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("[*] Setting up Trainer...")
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Set to False if your GPU doesn't support bfloat16
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    print("[*] Starting Training...")
    trainer.train()

    print(f"[*] Saving model to {NEW_MODEL}...")
    trainer.model.save_pretrained(NEW_MODEL)
    tokenizer.save_pretrained(NEW_MODEL)

    print("[*] Pushing to Hugging Face Hub...")
    # NOTE: You must be logged in via huggingface-cli for this to work
    try:
        trainer.model.push_to_hub(f"{HUGGINGFACE_USERNAME}/{NEW_MODEL}")
        tokenizer.push_to_hub(f"{HUGGINGFACE_USERNAME}/{NEW_MODEL}")
        print("[+] Successfully pushed to Hub!")
    except Exception as e:
        print(f"[-] Failed to push to Hub: {e}")

if __name__ == "__main__":
    main()
