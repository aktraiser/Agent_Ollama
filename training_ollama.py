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
    
    # Configuration recommandée par Unsloth
    dtype = None  # Auto-détection du dtype
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detection
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
    
    try:
        df = pd.read_csv(csv_file, sep=',')
        
        required_columns = ['content_type', 'title', 'questions', 'main_text', 'source', 'answers']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
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
            try:
                return {
                    "text": prompt_template.format(
                        content_type=str(row['content_type']),
                        title=str(row['title']),
                        question=str(row['questions']),
                        context=str(row['main_text']),
                        source=str(row['source']),
                        response=str(row['answers'])
                    )
                }
            except Exception as e:
                logger.error(f"Error formatting prompt for row: {e}")
                return {"text": ""}

        # Convert DataFrame to Dataset and apply formatting
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            format_prompt,
            remove_columns=dataset.column_names,
            desc="Formatting prompts"
        )

        if len(dataset) == 0:
            raise ValueError("Dataset is empty after processing")

        return dataset

    except Exception as e:
        logger.error(f"Error initializing dataset: {e}")
        raise

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

def save_model_for_ollama(model, tokenizer, output_dir="./ollama_export", push_to_hub=False, repo_id=None):
    """Export le modèle pour Ollama selon la documentation"""
    logger.info("Exporting model for Ollama...")
    
    # Créer le dossier d'export avec chemin absolu
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory at: {output_dir}")
    
    # Sauvegarder le modèle et le tokenizer
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving model and tokenizer to: {model_dir}")
    
    try:
        # Sauvegarder d'abord en format HF
        logger.info("Saving model in HF format...")
        model.save_pretrained(model_dir, safe_serialization=False)  # Désactiver safe_serialization pour éviter safetensors
        tokenizer.save_pretrained(model_dir)
        
        # Convertir en GGUF avec llama.cpp
        logger.info("Converting to GGUF format...")
        gguf_path = os.path.join(model_dir, "ggml-model-f16.gguf")
        
        # Utiliser convert.py de llama.cpp
        convert_cmd = [
            "python3",
            "-m", "llama_cpp.convert",
            "--outfile", gguf_path,
            "--outtype", "f16",
            model_dir
        ]
        
        subprocess.run(convert_cmd, check=True)
        logger.info(f"Model converted to GGUF: {gguf_path}")
        
        # Nettoyer les fichiers HF temporaires
        for f in os.listdir(model_dir):
            if f != "ggml-model-f16.gguf":
                os.remove(os.path.join(model_dir, f))
        
        logger.info("Cleaned up temporary files")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
    
    # Création du Modelfile avec le chemin relatif
    modelfile_content = '''FROM ./model/ggml-model-f16.gguf
TEMPLATE """{{ .Prompt }}"""
SYSTEM """Tu es un expert comptable français spécialisé dans le conseil aux entreprises. Réponds de manière précise et professionnelle."""
PARAMETER temperature 0.7
PARAMETER top_p 0.5
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER num_gpu_layers 0'''
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    logger.info(f"Created Modelfile at: {modelfile_path}")
    
    # Créer un README avec les instructions
    readme_content = """# Modèle Comptable Expert

Ce modèle a été entraîné pour répondre à des questions comptables.

## Installation locale

1. Copiez le dossier 'model' et le fichier 'Modelfile' sur votre machine locale
2. Ouvrez un terminal et naviguez vers le dossier contenant ces fichiers
3. Créez le modèle dans Ollama :
   ```bash
   ollama create comptable-expert -f Modelfile
   ```
4. Testez le modèle :
   ```bash
   ollama run comptable-expert "Quelle est la différence entre un bilan et un compte de résultat?"
   ```
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    logger.info(f"Created README at: {readme_path}")
    
    # Afficher la taille totale du dossier
    total_size = sum(os.path.getsize(os.path.join(dirpath,filename)) 
                    for dirpath, dirnames, filenames in os.walk(output_dir) 
                    for filename in filenames) / (1024 * 1024)  # Convert to MB
    
    logger.info(f"""
✨ Model successfully exported to {output_dir}!
📦 Total export size: {total_size:.2f} MB

To use this model locally:
1. Copy the following files to your local machine:
   - {os.path.join(output_dir, "model")}       (entire directory)
   - {os.path.join(output_dir, "Modelfile")}
   - {os.path.join(output_dir, "README.md")}    (contains installation instructions)
2. Follow the instructions in README.md to install and run the model
""")
    
    return output_dir

def setup_ollama():
    """Configure Ollama"""
    def is_ollama_installed():
        try:
            subprocess.run(['which', 'ollama'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    try:
        if not is_ollama_installed():
            logger.info("Ollama not found. Installing Ollama...")
            install_cmd = 'curl https://ollama.ai/install.sh | sh'
            subprocess.run(install_cmd, shell=True, check=True)
            logger.info("Ollama installed successfully")

        logger.info("Starting Ollama service...")
        subprocess.run(['ollama', 'serve'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      start_new_session=True)
        time.sleep(5)  # Attendre le démarrage
        return True
    except Exception as e:
        logger.error(f"Error with Ollama setup: {e}")
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
        
        # Export pour Ollama (local only by default)
        ollama_dir = save_model_for_ollama(
            model, 
            tokenizer,
            output_dir="./ollama_export",
            push_to_hub=False  # Set to True and provide repo_id to upload to HF Hub
        )
        
        if setup_ollama():
            logger.info("Creating Ollama model...")
            try:
                modelfile_path = os.path.join(ollama_dir, "Modelfile")
                # Ensure the Modelfile exists and is readable
                if not os.path.exists(modelfile_path):
                    raise FileNotFoundError(f"Modelfile not found at {modelfile_path}")
                
                # Create the model with explicit path
                result = subprocess.run(
                    ['ollama', 'create', 'comptable-expert', '-f', modelfile_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("""
✨ Model successfully exported and configured!

To use your model:
1. ollama run comptable-expert

For testing:
ollama run comptable-expert "Quelle est la différence entre un bilan et un compte de résultat?"
""")
                else:
                    logger.error(f"Failed to create model. Output: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create Ollama model: {e.stderr}")
            except Exception as e:
                logger.error(f"An error occurred while creating the model: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
