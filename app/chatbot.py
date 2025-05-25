import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned medical model
tokenizer = AutoTokenizer.from_pretrained("path_to_med_dialog_model")
model = AutoModelForCausalLM.from_pretrained("path_to_med_dialog_model")

def get_response(prompt):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = get_response(user_input)
    print(f"Chatbot: {response}")