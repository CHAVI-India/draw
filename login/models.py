from django.contrib.auth.models import AbstractUser
from django.db import models

class autoseg_user(models.Model):
    name = models.CharField(max_length=100)
    username = models.CharField(max_length=100)
    email = models.EmailField()

    def __str__(self):
        return self.username  # or any other field you want to represent the user

class Job(models.Model):
    time_stamp = models.CharField(max_length=20)  # Assuming time_stamp is a string in the format 'YYYY-MM-DD_HH-MM-SS'
    description = models.TextField()
    status = models.CharField(max_length=100)
    model_used = models.CharField(max_length=100)
    user = models.ForeignKey(autoseg_user, on_delete=models.CASCADE, related_name='jobs')

    def __str__(self):
        return self.description  # or any other field you want to represent the job



