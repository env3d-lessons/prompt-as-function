#!/bin/bash

grep -qxF "export LLAMA_CPP_LIB_PATH=/workspaces/$(basename $(pwd))/.devcontainer/" ~/.bashrc || \
echo "export LLAMA_CPP_LIB_PATH=/workspaces/$(basename $(pwd))/.devcontainer/" >> ~/.bashrc

grep -qxF 'export LD_LIBRARY_PATH=/workspaces/'"$(basename $(pwd))"'/.devcontainer/:$LD_LIBRARY_PATH' ~/.bashrc || \
echo 'export LD_LIBRARY_PATH=/workspaces/'"$(basename $(pwd))"'/.devcontainer/:$LD_LIBRARY_PATH' >> ~/.bashrc

export LLAMA_CPP_LIB=/workspaces/$(basename $(pwd))/.devcontainer/libllama.so
CMAKE_ARGS="-DLLAMA_BUILD=OFF" pip install llama-cpp-python==0.3.10

# wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
# smallest qwen model
wget -nc https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf
wget -nc https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q2_k.gguf
wget -nc https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q2_k.gguf 
wget -nc https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q2_k.gguf 

pip install openai

echo ""
echo "✅ DevContainer setup complete!"
echo "You can now start working on your assignment."
