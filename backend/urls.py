from django.urls import path
from . import views


urlpatterns = [
    path("", views.start),
    path("output", views.show),
]

