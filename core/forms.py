from django import forms

class DocumentForm(forms.Form):
    # Поле для разгрузки документа
    docfile = forms.FileField(
        label = 'Выберите документ для анализа'
    )


    question = forms.CharField(
        label='Задайте мне вопрос по документу',
        widget=forms.Textarea(attrs={'rows': '4'}),
        required=False  # <-- Добавим это для удобства
    )