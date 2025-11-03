import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# Импорты для продвинутого чанкинга
from langchain_community.document_loaders import UnstructuredFileLoader
import nltk

# Одноразовая проверка и загрузка данных NLTK при старте
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Скачивание данных NLTK (punkt)...")
    nltk.download('punkt')
    print("Данные NLTK успешно скачаны.")

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
print("Загрузка модели для эмбеддингов...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
print("Модель для эмбеддингов загружена.")

print("Загрузка LLM...")
LLM_ID = "Qwen/Qwen1.5-4B-Chat"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype="auto",
    device_map="auto"
)
print("LLM успешно загружена.")

print("Загрузка модели Re-ranker...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Re-ranker успешно загружен.")


# --- КОНЕЦ ЗАГРУЗКИ МОДЕЛЕЙ ---


def create_sentence_window_chunks(file_path, window_size=3):
    """
    Гибридный метод: Unstructured разделяет на блоки, NLTK - на предложения внутри блоков.
    """
    print(f"Загрузка и ИЕРАРХИЧЕСКАЯ обработка документа: {file_path}")
    loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast")
    documents = loader.load()

    all_sentences = []
    # Проходим по каждому логическому блоку от Unstructured
    for doc in documents:
        # И уже внутри блока делим на предложения с помощью NLTK
        all_sentences.extend(nltk.sent_tokenize(doc.page_content))

    chunks = []
    for i, sentence in enumerate(all_sentences):
        # Определяем границы "окна"
        start_index = max(0, i - window_size)
        end_index = min(len(all_sentences), i + 1 + window_size)

        # Создаем "окно" - соединяем предложения вокруг основного
        window = " ".join(all_sentences[start_index:end_index])

        # Сохраняем и основное предложение (для поиска), и окно (для контекста)
        chunks.append({
            'sentence': sentence,
            'window': window
        })

    print(f"Документ разделен на {len(chunks)} предложений/чанков.")
    return chunks


def add_sentence_chunks_to_collection(collection, chunks):
    """
    Индексирует отдельные предложения, сохраняя "окна" в метаданных.
    """
    sentences = [chunk['sentence'] for chunk in chunks]
    metadatas = [{'window': chunk['window']} for chunk in chunks]
    ids = [str(i) for i in range(len(sentences))]

    print("Создание эмбеддингов для отдельных предложений...")
    embeddings = embedding_model.encode(sentences, show_progress_bar=True)
    print("Эмбеддинги созданы.")

    print(f"Добавление {len(chunks)} чанков в коллекцию '{collection.name}'...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=sentences,
        metadatas=metadatas,
        ids=ids
    )
    print("Чанки успешно добавлены.")


def search_and_rerank(collection, query, top_k_retriever=30, top_k_reranker=5):
    """
    Двухэтапный поиск: быстрый поиск по векторам, затем точная пересортировка с Re-ranker'ом.
    """
    # Этап 1: Быстрый поиск по векторам для получения кандидатов
    query_embedding = embedding_model.encode(query)
    print(f"Поиск {top_k_retriever} кандидатов...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k_retriever,
        include=["documents", "metadatas"]
    )

    candidate_sentences = results["documents"][0]

    # Этап 2: Точная пересортировка с помощью Re-ranker'а
    print(f"Переранжирование {len(candidate_sentences)} кандидатов...")
    pairs = [[query, sentence] for sentence in candidate_sentences]
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    original_metadatas = results["metadatas"][0]
    sentence_scores_meta = list(zip(candidate_sentences, scores, original_metadatas))

    sentence_scores_meta.sort(key=lambda x: x[1], reverse=True)

    # Возвращаем "окна" от топ-N лучших результатов
    reranked_windows = [meta["window"] for sentence, score, meta in sentence_scores_meta[:top_k_reranker]]
    print(f"Топ-{top_k_reranker} контекстных 'окон' выбраны после переранжирования.")
    return reranked_windows


def generate_answer_from_context(context, query):
    prompt_template = f"""
    Инструкция: Ты — высокоточный ассистент для ответов на вопросы по документу.
    1.  Отвечай на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте.
    2.  Будь предельно кратким и точным. Не добавляй лишней информации.
    3.  Если в контексте нет ответа, ты ОБЯЗАН ответить: 'На основании предоставленного документа я не могу ответить на ваш вопрос'. Не выдумывай ничего.
    4.  ВАЖНО: Ответ должен быть дан СТРОГО на русском языке.

    Контекст:
    ---
    {" ".join(context)}
    ---

    Вопрос: {query}
    Краткий и точный ответ на русском языке:
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