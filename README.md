# source virtual environment from root folder:
` $> source ./.venv/bin/activate . `
# Restore environment from requirements file:
` $> pip install -r requirements.txt `

# Ollama Docker Compose Setup
Welcome to the Ollama Docker Compose Setup! This project simplifies the deployment of Ollama using Docker Compose, making it easy to run Ollama with all its dependencies in a containerized environment.
[![Star History Chart](https://api.star-history.com/svg?repos=valiantlynx/ollama-docker&type=Date)](https://star-history.com/#valiantlynx/ollama-docker&Date)

## Getting Started
### Prerequisites
Make sure you have the following prerequisites installed on your machine:
- Docker
- Docker Compose

#### GPU Support (Optional)
If you have a GPU and want to leverage its power within a Docker container, follow these steps to install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU integration
docker run --gpus all nvidia/cuda:11.5.2-base-ubuntu20.04 nvidia-smi
```

### Configuration

1. Clone the Docker Compose repository:

    ```bash
    git clone https://github.com/valiantlynx/ollama-docker.git
    ```

2. Change to the project directory:

    ```bash
    cd ollama-docker
    ```

## Usage

Start Ollama and its dependencies using Docker Compose:

if gpu is configured
```bash
docker-compose -f docker-compose-ollama-gpu.yaml up -d
```
else
```bash
docker-compose up -d
```

Visit [http://localhost:8000](http://localhost:8000) in your browser to access Ollama-webui.

### Model Installation

Navigate to settings -> model and install a model (e.g., llava-phi3). This may take a couple of minutes, but afterward, you can use it just like ChatGPT.

### Explore Langchain and Ollama

You can explore Langchain and Ollama within the project. A third container named **app** has been created for this purpose. Inside, you'll find some examples.

### Devcontainer and Virtual Environment

The **app** container serves as a devcontainer, allowing you to boot into it for experimentation. Additionally, the run.sh file contains code to set up a virtual environment if you prefer not to use Docker for your development environment.
if you have vs code and the `Remote Development´ extension simply opening this project from the root will make vscode ask you to reopen in container
## Stop and Cleanup

To stop the containers and remove the network:

```bash
docker-compose down
```

## Contributing

We welcome contributions! If you'd like to contribute to the Ollama Docker Compose Setup, please follow our [Contribution Guidelines](CONTRIBUTING.md).


## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it according to the terms of the license. Just give me a mention and some credit

## Contact

If you have any questions or concerns, please contact us at [vantlynxz@gmail.com](mailto:vantlynxz@gmail.com).

Enjoy using Ollama with Docker Compose! 🐳🚀



-------try new things:
https://ollama.com/hub/hub
- [OpenWebUI](https://docs.openwebui.com/ )
- [FAQ](https://github.com/ollama/ollama/blob/main/docs/README.md )^^how to steps as well^^
- [some scripts](https://github.com/CtrlAiDel/How-to-Run-Multiple-AI-Models-with-Ollama-and-Open-WebUI-in-Docker/blob/main/README.md )
- Microsoft Phi2 SLM: A small but very powerful model for coding
- As of 12/08/24, I was able to run this solution with NVIDIA GPU. Run following docker command to run this solution with CPUs.
    - $> docker compose -f docker-compose-ollama-gpu.yaml up. 
    - If GPU errors are recieved re-install GPU drivers for docker.
- [Llama model library](https://ollama.com/search ): Search here for ollama models to download.
- Nuget - Ollamasharp : https://www.nuget.org/packages/OllamaSharp/
    - Example w Semantic Kernel: https://dev.to/azure/extending-semantickernel-using-ollamasharp-for-chat-and-text-completion-4m10 
  - To download a model for Ollama, run following commands
      ```Shell
      $> docker exec -it ollama bash
       $> ollama list
       $> ollama pull <model name>:<size tag>
      ```
  - Open the Ollama WebUI from the firefox container
    - On Remote computer open browser and connect to the ubuntu server on port 5800 
        > e.g. http://ubuntu:5800 Or http:<Ubuntu Ip>:5800, assuming firewall is open.
    - Then in the firefox, type http://ollama-webui:8080 , this will open Ollama webui.
    - Now from left corner select model we like, or download new ones from above commands !!
- [Ollama config](https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored )
- [Examples](https://github.com/ollama/ollama-python/blob/main/examples/README.md)
- [Ollama on baremetal](https://www.restack.io/p/ollama-answer-download-model-offline-cat-ai )
- To this repo I have added a OllamaSharp example, basically a copy of the demo code from [OllamaSharp Git repo](https://github.com/awaescher/OllamaSharp ).
    - The demo folder under src contains this demo project, build it and run it, provide the ollama url, e.g. localhost:7869 when prompted.
    - If we run $> docker stats, we will see that the ollama model uses 800%+ CPUs.
    - Add more models, e.g. Phi2 and run the demo app to see what can it do.

