#!/bin/bash

# Met à jour pip
pip install --upgrade pip


# Installation d'unsloth avec la configuration qui fonctionne
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Installation des dépendances supplémentaires sans réinstaller les packages
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft pandas torch cmake accelerate bitsandbytes scikit-learn scipy joblib threadpoolctl

# Installation de llama-cpp-python pour la conversion GGUF
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Exécute le script Python
python training_ollama.py
