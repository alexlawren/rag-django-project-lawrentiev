# core/forms.py
from django import forms

class DocumentForm(forms.Form):
    # Явно указываем, что это поле НЕ обязательно
    docfile = forms.FileField(
        label='Выберите документ для анализа',
        required=False
    )
    # И это поле НЕ обязательно
    question = forms.CharField(
        label='Задайте ваш вопрос по документу',
        widget=forms.Textarea(attrs={'rows': 4}),
        required=False
    )