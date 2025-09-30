from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_ask, name='main_page'),
]