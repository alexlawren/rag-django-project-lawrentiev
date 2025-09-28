import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. --- ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ---
# Мы просто меняем ID модели на тот, который хотим запустить.
MODEL_ID = "Qwen/Qwen1.5-4B-Chat"

print(f"Начинаю загрузку модели: {MODEL_ID}")

# 2. Загрузка модели. 
#    Она автоматически загрузится на вашу видеокарту в 16-битном формате.
#    Это займет около 8 ГБ видеопамяти, что идеально для вашей карты.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Загружаем токенизатор для этой модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("="*50)
print("Модель успешно загружена и готова к работе! (Hugging Face Transformers)")
print("Введите 'выход' или 'exit' для завершения.")
print("="*50)

# 3. Основной цикл для общения
while True:
    user_input = input("\nВаш вопрос: ")
    if user_input.lower() in ["выход", "exit"]:
        break

    # 4. Подготовка промпта. 
    #    tokenizer сам знает, какой формат нужен для Qwen.
    messages = [
        {"role": "user", "content": user_input},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 5. Отправляем данные на GPU и генерируем ответ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )
    
    # 6. Декодируем и печатаем только сам ответ
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\nQwen: {response_text}")