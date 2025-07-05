import argparse
from huggingface_hub import login
import transformers
import torch

def download_and_save_model(model_id="nlpaueb/legal-bert-base-uncased", cache_dir="./.cache/legal-bert-base-uncased"):
    # Login to Hugging Face Hub
    login()

    # Download the model and tokenizer locally
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Save model & tokenizer
    model.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)
    print("Model and tokenizer have been saved to:", cache_dir)

def main():
    parser = argparse.ArgumentParser(description="Download and save a pretrained model from Hugging Face.")
    parser.add_argument("--model_id", type=str, help="Model ID from Hugging Face Hub", required=True)
    parser.add_argument("--cache_dir", type=str, help="Local directory to cache the model", required=True)
    
    args = parser.parse_args()
    download_and_save_model(args.model_id, args.cache_dir)

# if __name__ == "__main__":
#     main()


download_and_save_model(model_id="mistralai/Mistral-Nemo-Instruct-2407", cache_dir="./.cache/Mistral-Nemo-Instruct-2407")