from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms import ModelForm 

class LoginForm(forms.Form):
    username = forms.CharField(label='Username', max_length=100)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

class SignupForm(UserCreationForm):
    name = forms.CharField(max_length=100)

    class Meta:
        model = User
        fields = ['name', 'username', 'email', 'password1', 'password2']
        

