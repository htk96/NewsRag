from django.utils import timezone
from django.db import models

class News(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    rank = models.IntegerField()
    news_date = models.DateTimeField(default=timezone.now)
    news_body_title = models.CharField(max_length=255, blank=True, null=True)
    published_date = models.DateTimeField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title
    
class General_News(models.Model):
    press_name = models.CharField(max_length=30, default="")
    title = models.CharField(max_length=200) 
    url = models.URLField()  
    rank = models.IntegerField()  
    published_date = models.DateTimeField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)
    content = models.TextField(blank=True, null=True) 
    news_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.title

class A_News(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    rank = models.IntegerField()
    news_date = models.DateTimeField(default=timezone.now)
    content = models.TextField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title
    
class B_News(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    rank = models.IntegerField()
    news_date = models.DateTimeField(default=timezone.now)
    content = models.TextField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title
    
class C_News(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    rank = models.IntegerField()
    news_date = models.DateTimeField(default=timezone.now)
    content = models.TextField(blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title
    

class Summary(models.Model):
    news = models.ForeignKey(News, on_delete=models.CASCADE)
    summary_text = models.TextField()  
    created_at = models.DateTimeField(auto_now_add=True) 
    
class General_Summary(models.Model):
    news = models.ForeignKey(General_News, on_delete=models.CASCADE)
    summary_text = models.TextField()  
    created_at = models.DateTimeField(auto_now_add=True) 

class Interaction(models.Model):
    news = models.ForeignKey(News, on_delete=models.CASCADE, null=True, blank=True)
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question} - A: {self.answer}"