from django.utils import timezone
from django.db import models

class News(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    rank = models.CharField(max_length=8)
    news_date = models.DateTimeField(default=timezone.now)
    news_body_title = models.CharField(max_length=255, blank=True, null=True)
    published_date = models.DateTimeField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title


class Summary(models.Model):
    news = models.ForeignKey(News, on_delete=models.CASCADE)  # 관련 뉴스 기사
    summary_text = models.TextField()  # 요약된 텍스트 내용
    created_at = models.DateTimeField(auto_now_add=True)  # 요약 생성 날짜 및 시간

