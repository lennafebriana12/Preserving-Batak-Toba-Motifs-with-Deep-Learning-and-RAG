from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.chat_n8n, name='chat_n8n'), # <--- TAMBAHKAN BARIS INI
]