import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen1.5-4B-Chat"

print(f"Начинаю загрузку модели: {MODEL_ID}")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("="*50)
print("Модель успешно загружена и готова к работе! (Hugging Face Transformers)")
print("Введите 'выход' или 'exit' для завершения.")
print("="*50)

while True:
    user_input = input("\nВаш вопрос: ")
    if user_input.lower() in ["выход", "exit"]:
        break

    messages = [
        {"role": "user", "content": user_input},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\nQwen: {response_text}")