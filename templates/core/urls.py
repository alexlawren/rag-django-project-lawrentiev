from django.urls import path
from . import views

urlpatterns = [
    # Пустой путь '' означает главную страницу сайта.
    # views.upload_and_ask - это функция, которую нужно вызвать.
    # name='main_page' - это внутреннее имя для этого маршрута.
    path('', views.upload_and_ask, name='main_page'),
]