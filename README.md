# My LLM Project

This project runs a Hugging Face model locally.

## Setup
1. Create a virtual environment
2. Install dependencies
3. Run the model

## 1. Create a virtual environment
python -m venv llm-env

llm-env\Scripts\activate.bat (windows)

## 2. Install dependencies
pip install --upgrade pip

pip install torch transformers accelerate huggingface_hub

### Choose a Model from Hugging Face

Go to 👉 https://huggingface.co/models



Good beginner models:

distilgpt2 (very small, fast)

gpt2

TinyLlama/TinyLlama-1.1B-Chat-v1.0

mistralai/Mistral-7B-Instruct-v0.2 (needs strong hardware)

### Hardware rule of thumb
- 1–2 GB RAM → very small models
- 8 GB RAM → up to ~3B parameters
- 16+ GB RAM or GPU → larger models

### Where Are Models Stored Locally?
- Linux / macOS:
~/.cache/huggingface/

- Windows:
C:\Users\<username>\.cache\huggingface\

## Download Model Without Running It
```
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="distilgpt2",
    local_dir="./models/distilgpt2"
)
```

## Full code without warnings
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix: set pad token
tokenizer.pad_token = tokenizer.eos_token

prompt = "Once upon a time"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True
)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

