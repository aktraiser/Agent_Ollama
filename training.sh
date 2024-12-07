#!/bin/bash

# Création et activation de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances de base
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers==4.37.2
pip install peft==0.7.1
pip install trl==0.7.4
pip install pandas==2.1.4
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install scipy==1.11.4
pip install safetensors==0.4.1

# Installation d'unsloth
pip install unsloth[cu118] -U --user

# Installation d'Ollama (si nécessaire)
if ! command -v ollama &> /dev/null
then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Création du dossier de sortie si nécessaire
mkdir -p output

echo "Installation terminée. Vous pouvez maintenant lancer l'entraînement avec:"
echo "python training_ollama.py"
