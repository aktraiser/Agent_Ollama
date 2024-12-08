#!/bin/bash

# Met à jour pip
pip install --upgrade pip


# Installation d'unsloth avec la configuration qui fonctionne
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# Installation des dépendances supplémentaires sans réinstaller les packages
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft pandas torch cmake libcurl4 accelerate bitsandbytes scikit-learn scipy joblib threadpoolctl

# Exécute le script Python
python training_ollama.py
