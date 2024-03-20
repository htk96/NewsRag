from django.urls import reverse
from django.http import JsonResponse
from django.contrib import messages
from datetime import datetime

from django.shortcuts import render, get_object_or_404
import requests
from bs4 import BeautifulSoup
from django.utils import timezone
from .models import News

from .models import News, Summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
nltk.download('punkt')

def index(request):
    return render(request, 'index.html')


def crawl_news_view(request):
    today_news_exists = News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = 'https://www.aitimes.com/'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.select('div.auto-article div.item')

        for item in items:
            link_element = item.find('a')
            em_element = item.find('em')
            span_element = item.find('span')

            if link_element and em_element and span_element and em_element.text.strip():
                link = link_element['href']
                full_url = url + link
                rank = em_element.text.strip()
                title = span_element.text.strip()

                news_response = requests.get(full_url)
                news_soup = BeautifulSoup(news_response.text, 'html.parser')
                
                news_body_title = news_soup.select_one('.heading').text.strip()
                published_date_str = news_soup.select_one('li i.icon-clock-o').next_sibling.strip().replace('입력 ', '')
                published_date = datetime.strptime(published_date_str, '%Y.%m.%d %H:%M')
                content = ' '.join(p.text for p in news_soup.select('article#article-view-content-div p'))

                src_url = None
                iframe_element = news_soup.select_one('iframe')
                if iframe_element:
                    src_url = iframe_element['src']
                else:
                    img_element = news_soup.select_one('img')
                    if img_element:
                        src_url = img_element['src']

                News.objects.create(
                    title=title,
                    url=full_url,
                    rank=rank,
                    news_date=timezone.now(),
                    news_body_title=news_body_title,
                    published_date=published_date,
                    content=content,
                    photo_url=src_url
                )

    news_list = News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:10]
    today = datetime.now().strftime('%Y.%m.%d')
    messages.success(request, f'{today}일의 HOT 뉴스 업데이트')
    
    return render(request, 'index.html', {'news_list': news_list})


tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')
model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')

def summarize_text(request, news_id):
    news_article = get_object_or_404(News, pk=news_id)
    text_to_summarize = news_article.content  # 가정: News 모델에 'text' 필드가 있다고 가정합니다.
    
    prefix = "summarize: "
    inputs = tokenizer(prefix + text_to_summarize, return_tensors="pt", padding=True, truncation=True, max_length=512)
    summary_outputs = model.generate(**inputs, num_beams=4, max_length=200, early_stopping=True)
    summary_text = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

    # 요약을 데이터베이스에 저장합니다.
    summary, created = Summary.objects.get_or_create(
        news=news_article,
        defaults={'summary_text': summary_text}
    )
    if not created:
        summary.summary_text = summary_text
        summary.save()

    # 요약된 텍스트와 함께 페이지를 렌더링합니다.
    return JsonResponse({'summary': summary.summary_text})