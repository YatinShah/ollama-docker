Your dual Intel Xeon X5570 processors, 48GB of DDR3 RAM, and NVIDIA GeForce GTX 1050 Ti GPU form a solid foundation for a capable, budget-friendly AI agent server.
Because your GTX 1050 Ti has 4GB of VRAM, it cannot hold modern LLMs entirely in video memory. To make this setup work smoothly, we will configure hybrid CPU/GPU inference. Your 48GB of system RAM will host the bulk of the AI models, while your GPU offloads and accelerates processing.
------------------------------
## 🧠 Model Selection Strategy (Crucial for 4GB VRAM)
To avoid out-of-memory errors, you must use highly quantized, lightweight models.

* Primary Brain (LLM): llama3:8b-instruct-q4_K_M or qwen2.5:3b-instruct. The 3B Qwen model fits entirely in your 4GB VRAM for lightning-fast speeds. The 8B Llama model will split across your 48GB system RAM and GPU VRAM.
* Memory Translator (Embeddings): nomic-embed-text. This requires less than 500MB of RAM and is highly efficient.

------------------------------
## 🛠️ Hardware-Optimized Configuration
We will use native Ubuntu Server (bare metal) instead of Proxmox to maximize system resources and eliminate virtualizer overhead on this older architecture.
## 1. Update Drivers for the GTX 1050 Ti
Your Pascal-architecture card requires the legacy 535 proprietary driver for optimal stability with older CUDA versions.

sudo apt update && sudo apt install -y nvidia-driver-535-server nvidia-utils-535

Verify the installation by running nvidia-smi to ensure your 4GB frame buffer is recognized.
## 2. Install NVIDIA Container Toolkit
This allows Docker to pass inference math directly to your 1050 Ti Pascal cores.

curl -fsSL https://github.io | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://github.io | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

------------------------------
## 📦 Low-Overhead Software Stack
We will use Ollama, Qdrant (an incredibly lightweight C++ alternative to heavy Java vector stores), and Open WebUI (which features built-in RAG/memory pipeline management out of the box, saving you from running heavy Python frameworks like Langflow).
## Create docker-compose.yml

version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama_backend
    volumes:
      - ./ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_NUM_PARALLEL=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_memory
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: agent_ui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - VECTOR_DB=qdrant
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./webui_data:/app/backend/data
    depends_on:
      - ollama
      - qdrant
    restart: unless-stopped

Launch the pipeline:

docker compose up -d

------------------------------
## 🚀 Initialize the Models
Pull the optimized models tailored specifically for your 48GB RAM + 4GB VRAM matrix:

# Pull the 3B model (Fits perfectly into GTX 1050 Ti VRAM for fast execution)
docker exec -it ollama_backend ollama run qwen2.5:3b
# Pull the lightweight embedding model for the memory layers
docker exec -it ollama_backend ollama run nomic-embed-text

------------------------------
## 🧠 Enabling Persistent Agent Memory

   1. Open your browser to http://<your-server-ip>:3000.
   2. Go to Workspace -> Tools / Prompts to create your agent profile.
   3. Open Settings -> Documents / Memories.
   4. Set the embedding engine to Ollama and pick nomic-embed-text.
   5. When chatting with your custom agent, toggle on Memory. Open WebUI will silently embed your past chats, store them in the lightweight Qdrant database, and inject relevant history back into your Intel Xeon processors for contextual awareness.

Would you like help creating a systemd service script to make sure this entire stack boots up automatically if your server restarts?

## design choices around RAG and large documents.
Strategy 1 (Parent-Child): Best for complex text where deep context is needed. Searches tiny "child" fragments but passes the pre-linked "parent" block to the LLM.Trade-off: Increases Qdrant storage size due to text duplication.
Strategy 2 (ID Window Stitching): Best for continuous narratives (like books or logs). Fetches adjacent chunks (ID - 1 and ID + 1) to stitch context back together dynamically.Trade-off: Requires a secondary database read operation to fetch neighboring IDs.
Strategy 3 (Hybrid Document Store): Best for maximizing performance on older hardware. Keeps Qdrant ultra-lightweight by storing only math vectors and reference pointers, while keeping raw text on the filesystem or SQLite.Trade-off: Requires managing two distinct data storage layers in code.

