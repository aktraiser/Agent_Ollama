import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import time
import os
import logging
import subprocess
from pathlib import Path

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model(max_seq_length=2048):
    """Initialise le modèle selon les recommandations Unsloth"""
    logger.info("Loading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-détection
        load_in_4bit=True,  # Recommandé pour les GPU avec mémoire limitée
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    logger.info("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank pour LoRA
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,  # Optimisé
        bias="none",     # Optimisé
        use_gradient_checkpointing="unsloth",  # Pour le long contexte
        random_state=3407
    )
    
    return model, tokenizer

def initialize_dataset(tokenizer, csv_file):
    """Prépare le dataset avec le format de prompt standardisé"""
    logger.info(f"Loading dataset from {csv_file}")
    
    df = pd.read_csv(csv_file, sep=',')
    
    # Format de prompt Alpaca amélioré
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tu es un expert comptable spécialisé dans le conseil aux entreprises. Réponds à la question suivante concernant {content_type}: {title}

### Input:
{question}

Contexte:
{context}

Source: {source}

### Response:
{response}"""

    def format_prompt(row):
        return {
            "text": prompt_template.format(
                content_type=row['content_type'],
                title=row['title'],
                question=row['questions'],
                context=row['main_text'],
                source=row['source'],
                response=row['answers']
            )
        }
    
    dataset = Dataset.from_pandas(df).map(
        format_prompt,
        remove_columns=df.columns
    )
    
    return dataset

def train_model(model, tokenizer, dataset, max_seq_length):
    """Configure l'entraînement selon les recommandations"""
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=500
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args
    )
    
    return trainer.train()

def save_model_for_ollama(model, tokenizer, output_dir="./ollama_export"):
    """Export le modèle pour Ollama selon la documentation"""
    logger.info("Exporting model for Ollama...")
    
    # Sauvegarder d'abord le modèle LoRA
    logger.info("Saving LoRA adapter...")
    model.save_pretrained("lora_weights")
    
    # Export en GGUF
    logger.info("Converting to GGUF format...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Utilisation de q4_k_m pour un bon compromis taille/qualité
    model.push_to_hub_gguf(
        repo_id=output_dir,
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
        save_method="safetensors"
    )
    
    # Création du Modelfile
    modelfile_content = f'''FROM {os.path.abspath(output_dir)}

# Paramètres de génération
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### Response:"

# Template de chat personnalisé
TEMPLATE """{{{{.System}}}}

### Instruction:
{{{{.Prompt}}}}

### Input:
{{{{.Input}}}}

### Response:
{{{{.Response}}}}"""

# System prompt par défaut
SYSTEM """Tu es un expert comptable spécialisé dans le conseil aux entreprises. Tu dois fournir une réponse professionnelle et précise."""'''
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    logger.info(f"Modelfile created at: {modelfile_path}")
    return output_dir

def setup_ollama():
    """Configure Ollama"""
    try:
        subprocess.run(['ollama', 'serve'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      start_new_session=True)
        time.sleep(5)  # Attendre le démarrage
        return True
    except Exception as e:
        logger.error(f"Error starting Ollama: {e}")
        return False

if __name__ == "__main__":
    max_seq_length = 2048
    csv_file = "dataset2_comptable.csv"
    
    try:
        # Initialisation et entraînement
        model, tokenizer = initialize_model(max_seq_length)
        dataset = initialize_dataset(tokenizer, csv_file)
        train_model(model, tokenizer, dataset, max_seq_length)
        
        # Préparation pour l'export
        logger.info("Preparing model for inference...")
        model = FastLanguageModel.for_inference(model)
        
        # Export pour Ollama
        ollama_dir = save_model_for_ollama(model, tokenizer)
        
        if setup_ollama():
            logger.info("Creating Ollama model...")
            subprocess.run(['ollama', 'create', 'comptable-expert', '-f', 
                          os.path.join(ollama_dir, "Modelfile")])
            
            logger.info("""
✨ Model successfully exported and configured!

To use your model:
1. ollama run comptable-expert

For testing:
ollama run comptable-expert "Quelle est la différence entre un bilan et un compte de résultat?"
""")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
