from django.shortcuts import render
from .forms import DocumentForm

def upload_and_ask(request):
    form = DocumentForm()
    answer = None
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            question = form.cleaned_data['question']
            docfile = form.cleaned_data['docfile']

            answer = f"Отлично, вы загрузили файл '{docfile.name}' и задали вопрос '{question}'. Обработка в процессе разрабработки"
    return render(request, 'core/main_page.html', {'form': form, 'answer': answer})