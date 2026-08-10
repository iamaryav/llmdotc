import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the model
model_path = "./deepseek8"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with 8-bit quantization and CPU offloading
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,  # Use 8-bit quantization
        device_map="auto",  # Automatically offload to GPU/CPU
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
        max_memory={0: "10GB", "cpu": "12GB"}  # Limit GPU and CPU memory
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Set device
device = model.device
print(f"Using device: {device}")

# Input text
input_text = "What is the capital of India?"
print(f"My Question: {input_text}")

# Tokenize input and move to the correct device
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Limit output length to save memory
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.7
    )
except Exception as e:
    print(f"Error during generation: {e}")
    exit()

# Decode and print output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text: \n")
print(output_text)
