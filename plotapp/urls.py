from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('results/', views.all_results, name='all_results'),
    path("results/<str:run_id>/", views.view_result, name="view_result"),
    path("results/<str:run_id>/extract/", views.extracted_result, name="extracted_result"),
    path("results/<str:run_id>/delete/", views.delete_result, name="delete_result"),
    path('download/curve_csv/<str:filename>/', views.download_curve_csv, name='download_curve_csv'),
    path('download/png/<str:filename>/', views.download_png, name='download_png'),
    path('download/results_csv/<str:filename>/', views.download_results_csv, name='download_results_csv'),
]


