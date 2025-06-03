from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, target_dir):
    print(f"Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"Model {model_name} downloaded to {target_dir}")

if __name__ == "__main__":
    models = [
        "NousResearch/Llama-2-7b-chat-hf",     # Open LLaMA 2 Chat
        "tiiuae/falcon-7b",                    # Open Falcon 7B
        "EleutherAI/gpt-neo-2.7B"             # Smaller GPT-Neo
    ]
    for model in models:
        target = f"./local_models/{model.split('/')[-1]}"
        download_model(model, target)
