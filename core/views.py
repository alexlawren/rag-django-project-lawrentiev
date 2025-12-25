from django.shortcuts import render
from .forms import DocumentForm
from . import rag_logic
import chromadb
import hashlib
import os
from django.core.files.storage import FileSystemStorage

# Создаем постоянное подключение к векторной базе данных ChromaDB,
db_client = chromadb.PersistentClient(path="./chroma_db")


def main_page_view(request):
    # Создаем экземпляр нашей Django-формы.
    # request.POST or None - передает текстовые данные (вопрос), если они есть.
    # request.FILES or None - передает файловые данные (документ), если они есть.
    form = DocumentForm(request.POST or None, request.FILES or None)

    answer = None
    context = None

    # Пытаемся получить из сессии пользователя имя коллекции и имя документа,
    # с которыми он работал в прошлый раз. Сессия "помнит" состояние для каждого пользователя.
    collection_name = request.session.get('collection_name', None)
    document_name = request.session.get('document_name', None)

    # Главное условие: выполняем логику, только если пользователь нажал кнопку "Отправить"
    # (метод POST) и отправленные им данные прошли проверку (form.is_valid()).
    if request.method == 'POST' and form.is_valid():
        # Извлекаем "очищенные" данные из формы.
        uploaded_file = form.cleaned_data.get('docfile')
        question = form.cleaned_data.get('question')

        # СЦЕНАРИЙ 1: Пользователь загрузил новый документ
        if uploaded_file:
            # Создаем уникальное имя для коллекции в БД на основе MD5-хэша имени файла.
            # Это гарантирует, что для каждого файла будет своя уникальная база векторов.
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            collection_name = f"doc_{file_hash}"

            # Временно сохраняем загруженный файл на диск сервера, чтобы его можно было прочитать.
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_path = fs.path(filename)

            try:
                # 1. Отправляем файл на обработку в наш RAG-модуль для нарезки на чанки.
                chunks = rag_logic.create_sentence_window_chunks(uploaded_file_path)

                # Если чанки успешно созданы (документ не пустой).
                if chunks:
                    # 2. Создаем или получаем коллекцию в ChromaDB с уникальным именем.
                    collection = db_client.get_or_create_collection(name=collection_name)
                    # 3. Отправляем чанки на векторизацию и сохранение в эту коллекцию.
                    rag_logic.add_sentence_chunks_to_collection(collection, chunks)

                    # 4. "Запоминаем" в сессии, с какой коллекцией и документом теперь работает пользователь.
                    request.session['collection_name'] = collection_name
                    request.session['document_name'] = uploaded_file.name
                    document_name = uploaded_file.name  # Обновляем переменную для текущего отображения.

                    # Формируем сообщение об успехе для пользователя.
                    answer = f"Документ '{uploaded_file.name}' успешно обработан и сохранен."
                else:
                    # Если не удалось извлечь чанки, формируем ошибку.
                    answer = "Ошибка: не удалось обработать файл."
            finally:
                # 5. Независимо от результата, удаляем временный файл с диска.
                os.remove(uploaded_file_path)

        # === СЦЕНАРИЙ 2: Пользователь задал вопрос (файл не загружал) ===
        elif question:
            # Проверяем, "помнит" ли сессия, с каким документом мы работаем.
            if collection_name:
                # 1. Подключаемся к нужной коллекции в ChromaDB.
                collection = db_client.get_collection(name=collection_name)
                # 2. Выполняем двухэтапный поиск для получения контекста.
                context = rag_logic.search_and_rerank(collection, question)
                # 3. Генерируем ответ на основе найденного контекста и вопроса.
                answer = rag_logic.generate_answer_from_context(context, question)
            else:
                # Если в сессии нет информации о документе, просим пользователя сначала его загрузить.
                answer = "Ошибка: пожалуйста, сначала загрузите документ."

    # Финальный шаг: собираем HTML-страницу.
    # Передаем в шаблон 'main_page.html' все необходимые данные:
    # - form: объект формы для отображения полей ввода.
    # - answer: сгенерированный ответ или сервисное сообщение.
    # - context: список контекстных фрагментов.
    # - document_name: имя текущего активного документа.
    return render(request, 'core/main_page.html',
                  {'form': form, 'answer': answer, 'context': context, 'document_name': document_name})