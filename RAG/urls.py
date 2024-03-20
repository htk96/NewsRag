from django.urls import path
from . import views
from .views import (
    crawl_news_view,
)


urlpatterns = [
    path('', views.index, name='index'),
    path('crawl-news/', crawl_news_view, name='crawl-news'),
    path('summarize/<int:news_id>/', views.summarize_text, name='summarize_text'),
]
