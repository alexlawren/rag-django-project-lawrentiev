from django import forms

class DocumentForm(forms.ModelForm):
    # Поле для разгрузки документа
    docfile = forms.FileField(
        label = 'Выберите документ для анализа'
    )

    # Поле для ввода текстового запроса
    questions = forms.CharField(
        label = 'Задайте мне вопрос по документу',
        widget = forms.Textarea(attrs = {'rows':'4'}),
    )