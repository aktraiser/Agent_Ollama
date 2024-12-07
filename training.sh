#!/bin/bash
# Met à jour pip
pip install --upgrade pip

# Installe les dépendances nécessaires
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" "unsloth==2024.12.4" "transformers==4.36.2" "peft==0.7.1" "trl==0.7.10" "accelerate==0.21.0" "bitsandbytes==0.40.0" "pandas>=2.2.3" "scikit-learn" "scipy" "joblib"

# Execute the Python script
python training_ollama.py#!/bin/bash

# Configuration des couleurs pour les logs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Démarrage de la configuration de l'environnement...${NC}"

# Installation d'Ollama si nécessaire
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Installation d'Ollama...${NC}"
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Création des répertoires nécessaires
echo -e "${YELLOW}Création des répertoires...${NC}"
mkdir -p outputs
mkdir -p ollama_export
mkdir -p model_logs

# Démarrage d'Ollama en arrière-plan
echo -e "${YELLOW}Démarrage du serveur Ollama...${NC}"
ollama serve &

# Attente que le serveur Ollama soit prêt
sleep 5

echo -e "${GREEN}✨ Configuration terminée ! Démarrage de l'entraînement...${NC}"

# Lancement de l'entraînement
python training_ollama.py

# Nettoyage
trap 'pkill -f "ollama serve"' EXIT

echo -e "${GREEN}🎉 Processus terminé !${NC}"
