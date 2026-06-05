from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import ollama
import uvicorn

app = FastAPI(title="RoadWatch Ollama Proxy")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/generate")
async def generate(req: ChatRequest):
    try:
        # Call the local Ollama service running in the background
        response = ollama.chat(model='llama3.2:1b', messages=req.messages)
        content = response['message']['content']
        return {"reply": content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "Ollama proxy running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
