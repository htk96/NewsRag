# Generated by Django 5.0.2 on 2024-04-29 19:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RAG', '0010_alter_news_rank'),
    ]

    operations = [
        migrations.AlterField(
            model_name='news',
            name='rank',
            field=models.IntegerField(),
        ),
    ]
