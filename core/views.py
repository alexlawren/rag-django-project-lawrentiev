# core/views.py
from django.shortcuts import render
from .forms import DocumentForm
from . import rag_logic
import chromadb
import hashlib

db_client = chromadb.PersistentClient(path="./chroma_db")


def main_page_view(request):
    form = DocumentForm(request.POST or None, request.FILES or None)
    answer = None
    context = None

    collection_name = request.session.get('collection_name', None)
    document_name = request.session.get('document_name', None)

    if request.method == 'POST' and form.is_valid():
        uploaded_file = form.cleaned_data.get('docfile')
        question = form.cleaned_data.get('question')

        if uploaded_file:
            # Эта часть для загрузки файла остается без изменений
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            collection_name = f"doc_{file_hash}"
            document_text = rag_logic.get_document_text(uploaded_file)
            if document_text:
                text_chunks = rag_logic.get_text_chunks(document_text)
                collection = db_client.get_or_create_collection(name=collection_name)
                rag_logic.add_chunks_to_collection(collection, text_chunks)
                request.session['collection_name'] = collection_name
                request.session['document_name'] = uploaded_file.name
                document_name = uploaded_file.name
                answer = f"Документ '{uploaded_file.name}' успешно обработан и сохранен."
            else:
                answer = "Ошибка: не удалось прочитать текст из файла."

        elif question:
            if collection_name:
                collection = db_client.get_collection(name=collection_name)

                # --- ИЗМЕНЕННАЯ ЛОГИКА ---
                # 1. Сначала выполняем ШИРОКИЙ поиск, чтобы получить кандидатов
                candidate_docs = rag_logic.search_in_vector_store(collection, question)

                # 2. ВЫЗЫВАЕМ RE-RANKER, чтобы пересортировать кандидатов и выбрать лучших
                # Этот отфильтрованный список и будет нашим финальным контекстом
                context = rag_logic.rerank_documents(question, candidate_docs)
                # --- КОНЕЦ ИЗМЕНЕННОЙ ЛОГИКИ ---

                # 3. Генерируем ответ, используя уже улучшенный контекст
                answer = rag_logic.generate_answer_from_context(context, question)
            else:
                answer = "Ошибка: пожалуйста, сначала загрузите документ."

    return render(request, 'core/main_page.html',
                  {'form': form, 'answer': answer, 'context': context, 'document_name': document_name})