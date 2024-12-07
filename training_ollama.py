import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import time
import os
import logging
import subprocess
from pathlib import Path

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_model(max_seq_length):
    """Initialise le modèle avec les configurations optimales"""
    logger.info("Initializing model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-détection
        load_in_4bit=True,  # Quantification 4-bit pour l'efficacité
        attn_implementation="flash_attention_2",
        rope_scaling={"type": "dynamic", "factor": 2.0},
        trust_remote_code=True
    )
    
    # Configuration LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    return model, tokenizer

def initialize_dataset(tokenizer, csv_file):
    """Prépare le dataset pour l'entraînement"""
    logger.info(f"Loading dataset from {csv_file}")
    
    # Charger le CSV avec le bon séparateur
    df = pd.read_csv(csv_file, sep=';')
    
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tu es un expert comptable spécialisé dans le conseil aux entreprises. Tu dois fournir une réponse professionnelle et précise basée uniquement sur le contexte fourni.

### Input:
Type: {content_type}
Sujet: {title}
Document: {main_text}
Question: {questions}
Source: {source}

### Response:
{answers}"""

    EOS_TOKEN = tokenizer.eos_token or '</s>'
    
    def formatting_prompts_func(examples):
        texts = []
        for content_type, title, main_text, questions, answers, source in zip(
            examples["content_type"],
            examples["title"],
            examples["main_text"],
            examples["questions"],
            examples["answers"],
            examples["source"]
        ):
            text = prompt_template.format(
                content_type=content_type,
                title=title,
                main_text=main_text,
                questions=questions,
                answers=answers,
                source=source
            ) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    dataset = df.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=df.columns
    )
    
    return dataset

def train_model(model, tokenizer, dataset, max_seq_length):
    """Configure et lance l'entraînement"""
    logger.info("Starting model training...")
    
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1500,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args
    )
    
    return trainer.train()

def save_model_for_ollama(model, tokenizer, output_dir="./ollama_export"):
    """Exporte le modèle pour utilisation avec Ollama"""
    logger.info("Starting Ollama export process...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration des méthodes de quantification
    quantization_methods = [
        {"name": "q4_k_m", "enabled": True},  # Méthode recommandée
        {"name": "q8_0", "enabled": False},
        {"name": "q5_k_m", "enabled": False},
        {"name": "q3_k_m", "enabled": False}
    ]
    
    # Export GGUF
    for method in quantization_methods:
        if method["enabled"]:
            logger.info(f"Converting to GGUF format with {method['name']} quantization...")
            try:
                model.push_to_hub_gguf(
                    repo_id=output_dir,
                    tokenizer=tokenizer,
                    quantization_method=method["name"],
                    save_method="safetensors"
                )
                logger.info(f"Successfully exported model with {method['name']} quantization")
            except Exception as e:
                logger.error(f"Error during GGUF conversion with {method['name']}: {str(e)}")
                continue
    
    # Création du Modelfile
    modelfile_content = f"""
FROM {os.path.abspath(output_dir)}

# Paramètres de génération
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### Response:"

# Template de chat personnalisé
TEMPLATE """{{.System}}

### Instruction:
{{.Prompt}}

### Input:
{{.Input}}

### Response:
{{.Response}}"""

# System prompt par défaut
SYSTEM """Tu es un expert comptable spécialisé dans le conseil aux entreprises. Tu dois fournir une réponse professionnelle et précise basée uniquement sur le contexte fourni."""
"""
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    logger.info(f"\nModelfile created at: {modelfile_path}")
    
    return output_dir

def setup_ollama():
    """Configure et démarre Ollama"""
    try:
        # Vérifier si Ollama est installé
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if not result.stdout:
            logger.info("Installing Ollama...")
            subprocess.run(['curl', 'https://ollama.ai/install.sh', '|', 'sh'], check=True)
        
        # Démarrer le serveur Ollama
        logger.info("Starting Ollama server...")
        subprocess.Popen(['ollama', 'serve'])
        time.sleep(5)  # Attendre que le serveur démarre
        
        return True
    except Exception as e:
        logger.error(f"Error setting up Ollama: {str(e)}")
        return False

def create_ollama_model(model_dir, model_name="comptable-expert"):
    """Crée le modèle dans Ollama"""
    try:
        modelfile_path = os.path.join(model_dir, "Modelfile")
        subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path], check=True)
        logger.info(f"Successfully created Ollama model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error creating Ollama model: {str(e)}")
        return False

if __name__ == "__main__":
    # Configuration initiale
    max_seq_length = 2048
    csv_file = "dataset2_comptable.csv"  # Assurez-vous que ce fichier existe
    
    # Créer le dossier de logs
    os.makedirs("model_logs", exist_ok=True)
    file_handler = logging.FileHandler("model_logs/training.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    try:
        # Initialisation et entraînement
        logger.info("Starting training pipeline...")
        model, tokenizer = initialize_model(max_seq_length)
        dataset = initialize_dataset(tokenizer, csv_file)
        trainer_stats = train_model(model, tokenizer, dataset, max_seq_length)
        
        # Fusion du modèle
        logger.info("Merging model...")
        merged_model = model.merge_and_unload()
        
        # Export pour Ollama
        ollama_dir = save_model_for_ollama(merged_model, tokenizer)
        
        # Configuration d'Ollama
        if setup_ollama():
            if create_ollama_model(ollama_dir):
                logger.info("""
Modèle exporté et configuré avec succès!

Pour utiliser votre modèle:
1. ollama run comptable-expert

Pour tester une inférence:
ollama run comptable-expert "Quelle est la différence entre un bilan et un compte de résultat?"
""")
            
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")
        raise
    
    logger.info("Pipeline completed successfully!")
