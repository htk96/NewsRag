from django.http import JsonResponse
from django.contrib import messages
from datetime import datetime
import os
import pytz

from django.shortcuts import render, get_object_or_404
import requests
from bs4 import BeautifulSoup,NavigableString
from django.utils import timezone

from .models import News, Summary, A_News, B_News, C_News, General_News, General_Summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk
nltk.download('punkt')
from django.views.decorators.csrf import csrf_exempt


#---#
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler



def index(request):
    return render(request, 'index.html')

def laboratory(request):
    a_news_list = A_News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:5]
    b_news_list = B_News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:5]
    c_news_list = C_News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:5]
    
    context = {
        'a_news_list': a_news_list,
        'b_news_list': b_news_list,
        'c_news_list': c_news_list
    }
    
    return render(request, 'laboratory.html', context)

def chatbot(request):
    return render(request, 'chatbot.html')

def laboratory_RAG(request):
    return render(request, 'laboratory_RAG.html')

def General_Sum_Bot(request):
    today_date = timezone.now().date()
    news_list = General_News.objects.filter(news_date__date=today_date).order_by('press_name', 'rank')

    news_by_press = {}
    for news in news_list:
        if news.press_name not in news_by_press:
            news_by_press[news.press_name] = []
        news_by_press[news.press_name].append(news)

    return render(request, 'General_Sum_Bot.html', {'news_by_press': news_by_press})

def Ranking_Sum_ver1(request):
    news_list = News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:10]
    return render(request, 'Ranking_Sum_ver1.html', {'news_list': news_list})

def Ranking_RAG(request):
    news_list = News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:10]
    return render(request, 'Ranking_RAG.html', {'news_list': news_list})


# def crawl_news_view(request):
#     today_news_exists = News.objects.filter(news_date__date=timezone.now().date()).exists()

#     if not today_news_exists:
#         url = 'https://www.aitimes.com/'
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         items = soup.select('div.auto-article div.item')

#         for item in items:
#             link_element = item.find('a')
#             em_element = item.find('em')
#             span_element = item.find('span')

#             if link_element and em_element and span_element and em_element.text.strip():
#                 link = link_element['href']
#                 full_url = url + link
#                 rank = em_element.text.strip()
#                 title = span_element.text.strip()

#                 news_response = requests.get(full_url)
#                 news_soup = BeautifulSoup(news_response.text, 'html.parser')
                
#                 news_body_title = news_soup.select_one('.heading').text.strip()
#                 published_date_str = news_soup.select_one('li i.icon-clock-o').next_sibling.strip().replace('입력 ', '')
#                 published_date = datetime.strptime(published_date_str, '%Y.%m.%d %H:%M')
#                 content = ' '.join(p.text for p in news_soup.select('article#article-view-content-div p'))

#                 src_url = None
#                 figure_element = news_soup.select_one('figure.photo-layout.image')
#                 if figure_element:
#                     img_element = figure_element.find('img')
#                     if img_element and img_element.has_attr('src'):
#                         src = img_element['src']
#                         src_url = src if src.startswith('http') else url + src
#                 else:
#                     iframe_element = news_soup.select_one('iframe')
#                     if iframe_element:
#                         src_url = iframe_element['src']

#                 News.objects.create(
#                     title=title,
#                     url=full_url,
#                     rank=rank,
#                     news_date=timezone.now(),
#                     news_body_title=news_body_title,
#                     published_date=published_date,
#                     content=content,
#                     photo_url=src_url
#                 )
#                 A_News.objects.create(
#                     title=title,
#                     url=full_url,
#                     rank=rank,
#                     news_date=timezone.now(),
#                     content=content,
#                     photo_url=src_url
#                 )
#                 B_News.objects.create(
#                     title=title,
#                     url=full_url,
#                     rank=rank,
#                     news_date=timezone.now(),
#                     content=content,
#                     photo_url=src_url
#                 )
#                 C_News.objects.create(
#                     title=title,
#                     url=full_url,
#                     rank=rank,
#                     news_date=timezone.now(),
#                     content=content,
#                     photo_url=src_url
#                 )

#     news_list = News.objects.filter(news_date__date=timezone.now().date()).order_by('rank')[:10]
#     today = datetime.now().strftime('%Y.%m.%d')
#     messages.success(request, f'{today}일의 HOT 뉴스 업데이트')
    
#     return render(request, 'index.html', {'news_list': news_list})

def rag_news_view(request):
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
                figure_element = news_soup.select_one('figure.photo-layout.image')
                if figure_element:
                    img_element = figure_element.find('img')
                    if img_element and img_element.has_attr('src'):
                        src = img_element['src']
                        src_url = src if src.startswith('http') else url + src
                else:
                    iframe_element = news_soup.select_one('iframe')
                    if iframe_element:
                        src_url = iframe_element['src']

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

    news_list = News.objects.filter(news_date__date=timezone.now().date()).order_by('rank').prefetch_related('summary_set')
    today = timezone.now().strftime('%Y.%m.%d')
    messages.success(request, f'{today}일의 HOT 10 뉴스 업데이트')

    return render(request, 'Ranking_RAG.html', {'news_list': news_list})

# --- Summary 브러더들 --- #

""" 메인 요약 모델
이전 버전: custom4(encoder Selfattention만 LoRA Adaper를 적용해 Fine-Tuning한 모델)에서
현재 버전: custom1-3(Encoder, Decoder의 Selfattention & Feed Forward 모두에 LoRA Adapter를 적용해 Fine-Tuning한 모델)로 바꿈
> 달라진점: 전체 문맥의 해석이 좋아짐.
>> 학습의 차이: Trainable Params의 비율이 0.426 > 1.424로 많아짐. || All Params: 279,560,448(0.2B)
>>> 리소스 차이 1%의 Params의 증가로 GPU RAM 1.7GB 증가
"""
"""실험실 요약 모델
실험실의 모델은 Layer별 LoRA 적용의 차이를 확인하기 전 Hyperparams가 모델의 성능에 미치는 영향을 사용자 주관 평가로 비교하기 위해 구현함.
###따봉 기능을 넣어서 관리자가 사용자의 평가를 확인할 수 있게 추가해야겠음.####
> 학습량: 기존 모델 data_set=160000, batch=32, epoch=5 || 실험실 모델 main_data/32, batch=32, epoch=50, early_stop=True
"""
sum_model_name = 'eenzeenee/t5-base-korean-summarization'    
summarization_tokenizer  = AutoTokenizer.from_pretrained(sum_model_name)
summarization_model  = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
summarize_a = pipeline("summarization", model="du-kang/custom-scheduler-2", tokenizer="du-kang/custom-scheduler-2", max_length=3000)
summarize_b = pipeline("summarization", model="du-kang/custom-pre2", tokenizer="du-kang/custom-pre2", max_length=3000)
summarize_c = pipeline("summarization", model="du-kang/custom-pre3", tokenizer="du-kang/custom-pre3", max_length=3000)

def summarize_scheduler_a(request, news_id):
    return summarize_generic(request, news_id, A_News, summarize_a)

def summarize_scheduler_b(request, news_id):
    return summarize_generic(request, news_id, B_News, summarize_b)

def summarize_scheduler_c(request, news_id):
    return summarize_generic(request, news_id, C_News, summarize_c)

def summarize_generic(request, news_id, news_model, summarizer):
    news_article = get_object_or_404(news_model, pk=news_id) 
    text_to_summarize = news_article.content

    summary_outputs = summarizer(text_to_summarize, max_length=300, min_length=30, do_sample=False)

    summary_text = summary_outputs[0]['summary_text'] if summary_outputs else 'No summary generated.'

    return JsonResponse({'summary': summary_text})


def summarize_text(request, news_id):
    news_article = get_object_or_404(News, pk=news_id)
    
    try:
        summary = Summary.objects.get(news=news_article)
        
        return JsonResponse({'summary': summary.summary_text})
    except Summary.DoesNotExist:
        text_to_summarize = news_article.content
        
        prefix = "summarize: "
        inputs = summarization_tokenizer(prefix + text_to_summarize, return_tensors="pt", padding=True, truncation=True, max_length=3500)
        summary_outputs = summarization_model.generate(**inputs, num_beams=5, max_length=350, early_stopping=True)
        summary_text = summarization_tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

        summary, created = Summary.objects.get_or_create(
            news=news_article,
            defaults={'summary_text': summary_text}
        )
        if not created:
            summary.summary_text = summary_text
            summary.save()

        return JsonResponse({'summary': summary.summary_text})
    
def General_summarize_text(request, general_news_id):
    news_article = get_object_or_404(General_News, pk=general_news_id)
    
    try:
        summary = General_Summary.objects.get(news=news_article)
        return JsonResponse({'summary': summary.summary_text})
    except General_Summary.DoesNotExist:
        text_to_summarize = news_article.content
        
        prefix = "summarize: "
        inputs = summarization_tokenizer(prefix + text_to_summarize, return_tensors="pt", padding=True, truncation=True, max_length=3500)
        summary_outputs = summarization_model.generate(**inputs, num_beams=5, max_length=350, early_stopping=True)
        summary_text = summarization_tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

        summary, created = General_Summary.objects.get_or_create(
            news=news_article,
            defaults={'summary_text': summary_text}
        )
        if not created:
            summary.summary_text = summary_text
            summary.save()

        return JsonResponse({'summary': summary.summary_text})


# --- RAG 파트 --- #
def move_chat(request, news_id):
    news = get_object_or_404(News.objects.prefetch_related('summary_set'), pk=news_id)
    summary_text = news.summary_set.first().summary_text if news.summary_set.exists() else ""
    
    return render(request, 'chatbot.html', {'news': news, 'summary_text': summary_text}) 

def move_general_chat(request, news_id):
    news = get_object_or_404(General_News.objects.prefetch_related('general_summary_set'), pk=news_id)
    summary_text = news.general_summary_set.first().summary_text if news.general_summary_set.exists() else ""
    
    return render(request, 'general_chatbot.html', {'news': news, 'summary_text': summary_text}) 

@csrf_exempt
def rag_chat_view(request, news_id):
    
    os.getenv('LANGCHAIN_TRACING_V2')
    os.getenv('LANGCHAIN_PROJECT')
    os.getenv('LANGCHAIN_API_KEY')
    os.getenv('OPENAI_API_KEY')
    
    if request.method == 'POST':
        message = request.POST.get('message')
        news = get_object_or_404(News, pk=news_id)
        url_rag = news.url  

        article_strainer = bs4.SoupStrainer('article', id='article-view-content-div')

        loader = WebBaseLoader(
            web_paths=(url_rag,), 
            bs_kwargs={
                'parse_only': article_strainer 
            }
        )

        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")
        print(prompt)

        class StreamCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                print(token, end="", flush=True)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(message)

        final_response = f'"{response}"'
        return JsonResponse({'response': final_response})

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def general_rag_chat_view(request, news_id):
    
    if request.method == 'POST':
        message = request.POST.get('message')
        news = get_object_or_404(General_News, pk=news_id)
        url_rag = news.url  

        article_strainer = bs4.SoupStrainer('article', id='article-view-content-div')

        loader = WebBaseLoader(
            web_paths=(url_rag,), 
            bs_kwargs={
                'parse_only': article_strainer 
            }
        )

        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")
        print(prompt)

        class StreamCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                print(token, end="", flush=True)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(message)

        final_response = f'"{response}"'
        return JsonResponse({'response': final_response})

    return JsonResponse({'error': 'Invalid request'}, status=400)

#   --- 크롤링 녀석들 현재는 5개 사이트 --- #
# 크롤링 본체
def run_all_crawlers(request):
    crawl_joongang()  
    crawl_maeil()     
    crawl_yna()   
    crawl_seoul_shinmun()  
    crawl_aitimes()  
    today = datetime.now().strftime('%Y.%m.%d')
    messages.success(request, f'{today}일의 HOT 뉴스 업데이트')
    return render(request, 'index.html')

# AI타임즈
def crawl_aitimes():
    today_news_exists = News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = 'https://www.aitimes.com/'
        response = requests.get(url)
        if response.status_code == 200:
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
                    figure_element = news_soup.select_one('figure.photo-layout.image')
                    if figure_element:
                        img_element = figure_element.find('img')
                        if img_element and 'src' in img_element.attrs:
                            src = img_element['src']
                            src_url = src if src.startswith('http') else url + src

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
                    A_News.objects.create(
                        title=title,
                        url=full_url,
                        rank=rank,
                        news_date=timezone.now(),
                        content=content,
                        photo_url=src_url
                    )
                    B_News.objects.create(
                        title=title,
                        url=full_url,
                        rank=rank,
                        news_date=timezone.now(),
                        content=content,
                        photo_url=src_url
                    )
                    C_News.objects.create(
                        title=title,
                        url=full_url,
                        rank=rank,
                        news_date=timezone.now(),
                        content=content,
                        photo_url=src_url
                    )

# 중앙일보
def crawl_joongang():
    today_news_exists = General_News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = "https://www.joongang.co.kr/trend/daily"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("li", class_="card")[:5]
            rank = 1

            for article in articles:
                thumbnail = article.find("img")
                news_url = article.find("a")["href"]
                image_url = thumbnail["src"] if thumbnail else None

                news_response = requests.get(news_url)
                if news_response.status_code == 200:
                    news_soup = BeautifulSoup(news_response.content, "html.parser")
                    title = news_soup.find("h1", class_="headline").text.strip()
                    news_content = news_soup.find("div", class_="article_body").text.strip()

                    date_container = news_soup.find("time", itemprop="datePublished")
                    news_date = datetime.strptime(date_container["datetime"], '%Y-%m-%dT%H:%M:%S%z') if date_container else timezone.now()

                    news_entry = General_News(
                        press_name="중앙일보",
                        title=title,
                        url=news_url,
                        rank=rank,
                        published_date=news_date,
                        photo_url=image_url,
                        content=news_content,
                        news_date=timezone.now()  
                    )
                    news_entry.save()

                    rank += 1
        
# 매일신문
def crawl_maeil():
    today_news_exists = General_News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = "https://www.imaeil.com/"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            thumbnails = soup.find('div', class_='box wcms_bestnews_day').find_all('li')[:5]
            rank = 1
            
            for thumbnail in thumbnails:
                news_url = thumbnail.find('a')['href']
                title = thumbnail.find('a').text.strip()
                news_response = requests.get(news_url)
                if news_response.status_code == 200:
                    news_soup = BeautifulSoup(news_response.text, 'html.parser')
                    
                    image_tag = news_soup.find('div', class_='article_content').find('img')
                    photo_url = image_tag['src'] if image_tag else None
                    
                    content = ""
                    article_paragraphs = news_soup.find('div', class_='article_content').find_all('p')
                    for paragraph in article_paragraphs:
                        content += paragraph.get_text(strip=True) + "\n"

                    date_container = news_soup.find('span', class_='pblsh_time') 
                    if date_container:
                        published_date = datetime.strptime(date_container.text.strip(), '%Y-%m-%d %H:%M')  
                        published_date = pytz.timezone('Asia/Seoul').localize(published_date) 
                    else:
                        published_date = timezone.now()  

                    # Save to database
                    news_entry = General_News(
                        press_name="매일신문",
                        title=title,
                        url=news_url,
                        rank=rank,
                        published_date=published_date,
                        photo_url=photo_url,
                        content=content,
                        news_date=timezone.now()
                    )
                    news_entry.save()
                    
                    rank += 1
        
# 연합뉴스
def crawl_yna():
    today_news_exists = General_News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = "https://www.yna.co.kr/theme/topnews-history"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("div", class_="item-box01")
            
            for rank, article in enumerate(articles[:5], start=1):
                link = article.find("a")["href"].replace("//", "https://")
                response = requests.get(link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    title = soup.find("h1", class_="tit").text.strip()

                    date_container = soup.find("p", class_="update-time")
                    try:
                        published_date = datetime.strptime(date_container.text.strip(), '%Y-%m-%d %H:%M') 
                        published_date = pytz.timezone('Asia/Seoul').localize(published_date)
                    except ValueError:
                        published_date = timezone.now() 
                    
                    photo_url = soup.find("meta", property="og:image")["content"].replace("//", "https://")
                    content = ""
                    article_paragraphs = soup.find("article", class_="story-news article").find_all('p')
                    for paragraph in article_paragraphs:
                        content += paragraph.get_text(strip=True) + "\n"
                    
                    # Save to database
                    news_entry = General_News(
                        press_name="연합뉴스",
                        title=title,
                        url=link,
                        rank=rank,
                        published_date=published_date,
                        photo_url=photo_url,
                        content=content,
                        news_date=timezone.now()
                    )
                    news_entry.save()
        
# 서울신문
def crawl_seoul_shinmun():
    today_news_exists = General_News.objects.filter(news_date__date=timezone.now().date()).exists()

    if not today_news_exists:
        url = "https://www.seoul.co.kr/"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_list = soup.select('.topRankNews .topRankList li')
            
            for rank, news in enumerate(news_list, start=1):
                news_link = news.find('a')['href']
                full_news_link = url + news_link if news_link.startswith('/') else news_link
                
                news_response = requests.get(full_news_link)
                if news_response.status_code == 200:
                    news_soup = BeautifulSoup(news_response.content, 'html.parser')

                    title_div = news_soup.find('div', class_='articleTitle')
                    title = title_div.find('h1').text.strip() if title_div and title_div.find('h1') else 'Title not found'
                    
                    image_element = news_soup.find('div', class_='expendImageWrap')
                    image_link = image_element.find('img')['src'] if image_element and image_element.find('img') else None
                    
                    date_info = news_soup.find('span', class_='writeInfo')
                    if date_info:
                        try:
                            published_date = datetime.strptime(date_info.text.strip(), '%Y-%m-%d %H:%M') 
                            published_date = pytz.timezone('Asia/Seoul').localize(published_date) 
                        except ValueError:
                            published_date = timezone.now() 
                    else:
                        published_date = timezone.now()
                    
                    content_div = news_soup.find('div', class_='viewContent body18 color700')
                    content = ""
                    if content_div:
                        for element in content_div.contents:
                            if isinstance(element, NavigableString):
                                text = str(element).strip()
                                if text:
                                    content += text + " "
                            elif element.name == 'br':
                                content += "\n"

                    news_entry = General_News(
                        press_name="서울신문",
                        title=title,
                        url=full_news_link,
                        rank=rank,
                        published_date=published_date,
                        photo_url=image_link,
                        content=content,
                        news_date=timezone.now() 
                    )
                    news_entry.save()