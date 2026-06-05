FROM python:3.10-slim

WORKDIR /app

# Install curl and other necessary tools
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-pull the model during build so it starts instantly on wake
RUN bash -c "ollama serve & sleep 5 && ollama pull llama3.2:1b && kill \$!"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY start.sh .
RUN chmod +x start.sh

# Expose port (HF Spaces use 7860 by default)
EXPOSE 7860

# Command to run the setup script
CMD ["./start.sh"]
