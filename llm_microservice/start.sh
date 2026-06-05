#!/bin/bash

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
while ! curl -s http://127.0.0.1:11434 > /dev/null; do
    sleep 1
done

echo "Ollama is running. Pulling the model..."
# Pull the requested model
ollama pull llama3.2:1b

echo "Model pulled successfully. Starting FastAPI..."
# Start the FastAPI server (which acts as a proxy on port 7860)
uvicorn main:app --host 0.0.0.0 --port 7860
