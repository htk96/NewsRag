# Generated by Django 5.0.2 on 2024-04-29 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RAG', '0009_interaction'),
    ]

    operations = [
        migrations.AlterField(
            model_name='news',
            name='rank',
            field=models.IntegerField(max_length=8),
        ),
    ]
