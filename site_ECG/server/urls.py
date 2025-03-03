from django.urls import path
from .views import upload_data

urlpatterns = [
    path("", upload_data, name="upload"),
]