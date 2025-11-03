from django.shortcuts import render
from .forms import DocumentForm
from . import rag_logic
import chromadb
import hashlib
import os
from django.core.files.storage import FileSystemStorage

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
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            collection_name = f"doc_{file_hash}"

            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_path = fs.path(filename)

            try:
                chunks = rag_logic.create_sentence_window_chunks(uploaded_file_path)

                if chunks:
                    collection = db_client.get_or_create_collection(name=collection_name)
                    rag_logic.add_sentence_chunks_to_collection(collection, chunks)

                    request.session['collection_name'] = collection_name
                    request.session['document_name'] = uploaded_file.name
                    document_name = uploaded_file.name
                    answer = f"Документ '{uploaded_file.name}' успешно обработан и сохранен."
                else:
                    answer = "Ошибка: не удалось обработать файл."
            finally:
                os.remove(uploaded_file_path)

        elif question:
            if collection_name:
                collection = db_client.get_collection(name=collection_name)
                context = rag_logic.search_and_rerank(collection, question)
                answer = rag_logic.generate_answer_from_context(context, question)
            else:
                answer = "Ошибка: пожалуйста, сначала загрузите документ."

    return render(request, 'core/main_page.html',
                  {'form': form, 'answer': answer, 'context': context, 'document_name': document_name})