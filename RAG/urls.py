from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('Ranking_Sum_ver1', views.Ranking_Sum_ver1, name='Ranking_Sum_ver1'),
    path('Ranking_RAG', views.Ranking_RAG, name='Ranking_RAG'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('General_Sum_Bot', views.General_Sum_Bot, name='General_Sum_Bot'),
    path('laboratory', views.laboratory, name='laboratory'),
    path('laboratory_RAG', views.laboratory_RAG, name='laboratory_RAG'),
    # --- 함수들 ---
    # path('crawl-news/', views.crawl_news_view, name='crawl-news'),
    path('run_all_crawlers/', views.run_all_crawlers, name='run_all_crawlers'),
    path('rag_news_view', views.rag_news_view, name='rag_news_view'),
    path('summarize/<int:news_id>/', views.summarize_text, name='summarize_text'),
    path('General_summarize/<int:general_news_id>/', views.General_summarize_text, name='General_summarize'),
    path('chatbot/<int:news_id>/', views.move_chat, name='move_chat'),
    path('general_chatbot/<int:news_id>/', views.move_general_chat, name='move_general_chat'),
    path('rag_chat_view/<int:news_id>/', views.rag_chat_view, name='rag_chat_view'),
    path('general_rag_chat_view/<int:news_id>/', views.general_rag_chat_view, name='general_rag_chat_view'),
    path('summarize-a/<int:news_id>/', views.summarize_scheduler_a, name='summarize-a'),
    path('summarize-b/<int:news_id>/', views.summarize_scheduler_b, name='summarize-b'),
    path('summarize-c/<int:news_id>/', views.summarize_scheduler_c, name='summarize-c'),
    # path('rag_search/', views.rag_search_view, name='rag_search'),
    # path('news/<int:news_id>/ask/', views.news_rag_view, name='news_rag_view'),
    #       받아주고, 불러오고, 호출되고
]
