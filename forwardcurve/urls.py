from django.contrib import admin
from django.urls import path, include
from django.conf.urls import handler404, handler500
from plotapp import views

handler404 = views.custom_404 # noqa: F811
handler500 = views.custom_500 # noqa: F811

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('plotapp.urls')),
]
