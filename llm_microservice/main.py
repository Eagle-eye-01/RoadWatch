import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="RoadWatch LLM Microservice")

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global _tokenizer, _model, _device
    if _model is None:
        model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        print(f"[*] Loading {model_id} on {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if _device == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        ).to(_device)
    print("[*] Model loaded successfully.")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/generate")
async def generate(req: ChatRequest):
    prompt = _tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    
    with torch.no_grad():
        outputs = _model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    content = _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {"reply": content}

@app.get("/")
def health():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
