from transformers import pipeline

print("Loading model...")
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct", device="cpu")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

print("Generating...")
res = pipe(messages, max_new_tokens=50)
print(res[0]["generated_text"][-1]["content"])
