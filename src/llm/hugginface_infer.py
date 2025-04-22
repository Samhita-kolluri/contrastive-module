from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

MODEL_DIR = "model/stance-huggingface"

# Load tokenizer (disable fast tokenizer to avoid tokenizer.json bug)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_contrastive_outputs(premise, num_return_sequences=3):
    prompt = f"Premise: {premise}\nHypothesis:"
    outputs = generator(
        prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    return [out["generated_text"].split("Hypothesis:")[-1].strip() for out in outputs]

if __name__ == "__main__":
    test_input = "Technology will improve democratic participation."
    results = generate_contrastive_outputs(test_input)
    print("\nGenerated Contrastive Hypotheses:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")
