# Agent Ollama Setup Guide

This guide will help you set up and run the training environment on a VM and deploy locally with Ollama.

## VM Setup & Training

### 1. Download Private Key
After launching your VM instance, download the provided private key for secure access.

### 2. Set Key Permissions
Open your terminal and set the correct permissions:
```bash
chmod 400 private_key.pem
```

### 3. Connect to VM
Use SSH to connect:
```bash
ssh root@38.128.233.210 -p 1234 -i private_key.pem
```

### 4. Clone Repository
Once connected to the VM:
```bash
git clone https://github.com/aktraiser/Agent_Ollama.git
cd Agent_Ollama
```

### 5. Prepare Training Data
Ensure your dataset (dataset2_comptable.csv) is in the correct format with columns:
- content_type
- title
- questions
- main_text
- source
- answers

### 6. Run Training
Start the training process:
```bash
python training_ollama.py
```

## Training Monitoring

The training script displays:
- Loss values at each step
- Gradient norm
- Learning rate
- Training samples per second
- Total training time

Model specifications:
- Base model: Meta-Llama-3.1-8B
- Method: LoRA with Unsloth
- Batch size: 2 per device
- Gradient Accumulation: 4 steps
- Total batch size: 8

# Local Deployment with Ollama

## 1. On Your Local Machine

### Check Directory Structure
```bash
ls -lh ./ollama_export/
du -sh ./ollama_export/
```

### Set Key Permissions
```bash
chmod 400 private_key.pem
```

### Copy Model from VM
```bash
scp -P 1234 -i private_key.pem -r root@38.128.233.210:~/Agent_Ollama/ollama_export .
```

## 2. Install Ollama

### For macOS
1. Download Ollama:
   - Visit https://ollama.com/download/mac
   - Or use direct link in browser
2. Install:
   - Open the downloaded .dmg file
   - Drag Ollama to Applications folder
   - Launch Ollama from Applications

## 3. Deploy Model

### Create Model
```bash
cd ollama_export
ollama create comptable-expert -f Modelfile
```

### Test Model
```bash
ollama run comptable-expert "Quelle est la différence entre un bilan et un compte de résultat?"
```

## Model Information

- Export size: ~177MB (LoRA adapters only)
- Base model: Downloaded automatically by Ollama
- Location: ./ollama_export/
  - model/ (LoRA adapters)
  - Modelfile (Ollama configuration)
  - README.md (Installation instructions)

## Support

For any issues:
1. Check Ollama logs
2. Verify Ollama service is running
3. Ensure all model files are present
4. Contact support team if needed
