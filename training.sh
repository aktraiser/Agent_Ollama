#!/bin/bash

# Met à jour pip
pip install --upgrade pip


# Installation d'unsloth avec la configuration qui fonctionne
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Installation des dépendances supplémentaires sans réinstaller les packages
pip install --no-deps "xformers<0.0.26" "trl<0.9.0"


# Installation de torch et triton d'abord
pip install torch==2.0.1
pip install triton==2.0.0

# Installation des dépendances de base d'abord
pip install torch==2.0.1
pip install transformers==4.36.2
pip install accelerate==0.21.0
pip install bitsandbytes==0.40.0
pip install pandas>=2.2.3
pip install scikit-learn
pip install scipy


# Exécute le script Python
python training_ollama.py
