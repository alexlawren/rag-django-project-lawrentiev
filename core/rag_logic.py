import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
print("Загрузка модели для эмбеддингов...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
print("Модель для эмбеддингов загружена.")

print("Загрузка LLM... Это может занять некоторое время.")
LLM_ID = "Qwen/Qwen1.5-4B-Chat"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype="auto",
    device_map="auto"
)
print("LLM успешно загружена.")

# --- НОВЫЙ БЛОК: ЗАГРУЗКА RE-RANKER ---
print("Загрузка модели Re-ranker...")
# Мы используем легкую, но очень эффективную модель-кросс-энкодер
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Re-ranker успешно загружен.")


# --- КОНЕЦ НОВОГО БЛОКА ---


# Функции get_document_text, get_text_chunks, add_chunks_to_collection остаются БЕЗ ИЗМЕНЕНИЙ
def get_document_text(uploaded_file):
    text = ""
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    if file_extension == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == 'docx':
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_extension == 'txt':
        text = uploaded_file.read().decode("utf-8")
    else:
        return None
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def add_chunks_to_collection(collection, text_chunks):
    """Векторизует чанки и добавляет их в существующую коллекцию ChromaDB."""
    print("Создание эмбеддингов для фрагментов текста...")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    print("Эмбеддинги созданы.")
    chunk_ids = [str(i) for i in range(len(text_chunks))]
    print(f"Добавление {len(text_chunks)} фрагментов в коллекцию '{collection.name}'...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=text_chunks,
        ids=chunk_ids
    )
    print("Фрагменты успешно добавлены.")


def search_in_vector_store(collection, query):
    query_embedding = embedding_model.encode(query)

    # --- ИЗМЕНЕНИЕ: Увеличиваем количество кандидатов для Re-ranker ---
    # Мы берем больше документов (20), чтобы дать ре-ранкеру больше выбора.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20
    )
    return results['documents'][0]


# --- НОВАЯ ФУНКЦИЯ: RERANK_DOCUMENTS ---
def rerank_documents(query, documents, top_n=5):
    """
    Пересортировывает документы, используя модель CrossEncoder, и возвращает top_n лучших.
    """
    print(f"Переранжирование {len(documents)} документов...")
    # Создаем пары [вопрос, документ] для модели
    pairs = [[query, doc] for doc in documents]

    # Получаем оценки релевантности от модели
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    # Соединяем документы с их оценками
    doc_scores = list(zip(documents, scores))

    # Сортируем по оценке в порядке убывания
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Возвращаем только текст top_n лучших документов
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]
    print(f"Топ-{top_n} документов после переранжирования выбраны.")
    return reranked_docs


# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---


# Ваша исходная функция generate_answer_from_context остается БЕЗ ИЗМЕНЕНИЙ
def generate_answer_from_context(context, query):
    """
    Генерирует ответ на вопрос, используя найденный контекст.
    """
    prompt_template = f"""
    Инструкция: Ты — высокоточный ассистент для ответов на вопросы по документу.
    1.  Отвечай на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте.
    2.  Будь предельно кратким и точным. Не добавляй лишней информации, которая не требуется для ответа.
    3.  Если вопрос требует двух фактов, предоставь только эти два факта.
    4.  Если в контексте нет ответа, ты ОБЯЗАН ответить: 'На основании предоставленного документа я не могу ответить на ваш вопрос'. Не выдумывай ничего.

    Контекст:
    ---
    {" ".join(context)}
    ---

    Вопрос: {query}
    Краткий и точный ответ:
    """
    messages = [{"role": "user", "content": prompt_template}]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Генерация ответа с помощью LLM...")
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1
    )
    response_text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Ответ сгенерирован.")
    return response_text