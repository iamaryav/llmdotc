import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the model
model_path = "./deepseek"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with FP16 precision and offload to CPU/GPU
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
        device_map="auto" if torch.cuda.is_available() else "cpu",  # Use GPU if available
        low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Set device
device = model.device
print(f"Using device: {device}")

# Input text
# input_text = "Hi, How are you?"
q_1 = "2 + 2 - 3"
print(f"My Question: {q_1}")

# Tokenize input and move to the correct device
inputs = tokenizer(q_1, return_tensors="pt").to(device)

# Generate output
try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
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