#!/bin/bash
# Met à jour pip
pip install --upgrade pip

# Installation des dépendances avec des versions spécifiques
pip install transformers==4.36.2
pip install trl==0.7.4
pip install peft==0.7.1
pip install accelerate==0.21.0
pip install bitsandbytes==0.40.0
pip install pandas>=2.2.3
pip install scikit-learn
pip install scipy
pip install xformers==0.0.22

# Installation d'unsloth après les dépendances
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Exécute le script Python
python training_ollama.py
