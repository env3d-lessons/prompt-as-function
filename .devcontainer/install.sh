#!/bin/bash

grep -qxF "export LLAMA_CPP_LIB_PATH=/workspaces/$(basename $(pwd))/.devcontainer/" ~/.bashrc || \
echo "export LLAMA_CPP_LIB_PATH=/workspaces/$(basename $(pwd))/.devcontainer/" >> ~/.bashrc

grep -qxF 'export LD_LIBRARY_PATH=/workspaces/'"$(basename $(pwd))"'/.devcontainer/:$LD_LIBRARY_PATH' ~/.bashrc || \
echo 'export LD_LIBRARY_PATH=/workspaces/'"$(basename $(pwd))"'/.devcontainer/:$LD_LIBRARY_PATH' >> ~/.bashrc

export LLAMA_CPP_LIB=/workspaces/$(basename $(pwd))/.devcontainer/libllama.so
CMAKE_ARGS="-DLLAMA_BUILD=OFF" pip install llama-cpp-python==0.3.10

pip install openai

echo ""
echo "✅ DevContainer setup complete!"
echo "You can now start working on your assignment."
