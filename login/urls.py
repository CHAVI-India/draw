from django.urls import path
from . import views

urlpatterns = [
    path('home/',views.home,name='home'),
    path('', views.signup_view, name='signup_view'),
    path('login/',views.login_view,name='login_view'),
    path('signup/',views.signup_view,name='signup_view'),
     path('logout/', views.logout_view, name='logout'),
     path('upload_images/', views.upload_images, name='upload_images'),
     path('results/', views.results, name='results'),
     path('show_image/<int:job_id>/<str:file_name>/', views.image_slider_view, name='image_slider_view'),
    path('download/<int:job_id>/<str:file_name>/', views.download_file, name='download_file'),
    path('download_fol/<int:job_id>/<str:file_name>/', views.download_folder, name='download_folder'),
    path('job_result/<int:job_id>/', views.job_result, name='job_result'),
    
]