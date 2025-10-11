# core/views.py
from django.shortcuts import render
from .forms import DocumentForm
from . import rag_logic


def main_page_view(request):
    form = DocumentForm(request.POST or None, request.FILES or None)
    answer = None
    context = None
    document_name = request.session.get('document_name', None)

    if request.method == 'POST' and form.is_valid():
        uploaded_file = form.cleaned_data.get('docfile')
        question = form.cleaned_data.get('question')

        # Переменная для хранения чанков в текущем запросе
        current_chunks = None

        # --- НОВАЯ УЛУЧШЕННАЯ ЛОГИКА ---

        # Сначала ВСЕГДА обрабатываем новый файл, если он есть
        if uploaded_file:
            print(f"Обработка нового файла: {uploaded_file.name}")
            document_text = rag_logic.get_document_text(uploaded_file)

            if document_text:
                current_chunks = rag_logic.get_text_chunks(document_text)
                request.session['text_chunks'] = current_chunks
                request.session['document_name'] = uploaded_file.name
                document_name = uploaded_file.name
                answer = f"Документ '{uploaded_file.name}' успешно обработан. "
            else:
                answer = "Ошибка: не удалось прочитать текст из файла."
                # Очищаем сессию в случае ошибки
                request.session.pop('text_chunks', None)
                request.session.pop('document_name', None)
        else:
            # Если новый файл не загружен, берем чанки из сессии
            current_chunks = request.session.get('text_chunks')

        # ТЕПЕРЬ, после обработки файла, проверяем, был ли задан вопрос
        if question:
            if current_chunks:
                print(f"Выполнение поиска по вопросу: '{question}'")
                # Создаем векторную базу "на лету" из актуальных чанков
                vector_store = rag_logic.create_vector_store(current_chunks)

                if vector_store:
                    context = rag_logic.search_in_vector_store(vector_store, question)
                    # Если до этого уже было сообщение, добавляем к нему. Иначе - создаем новое.
                    if answer:
                        answer += "Вот результаты поиска по вашему вопросу:"
                    else:
                        answer = "Вот наиболее релевантные фрагменты по вашему вопросу:"
                else:
                    answer = "Произошла ошибка при создании векторной базы."
            else:
                answer = "Ошибка: пожалуйста, сначала загрузите документ, чтобы задать по нему вопрос."

    return render(
        request,
        'core/main_page.html',
        {
            'form': form,
            'answer': answer,
            'context': context,
            'document_name': document_name,
        }
    )