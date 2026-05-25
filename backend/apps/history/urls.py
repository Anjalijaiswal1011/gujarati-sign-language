from django.urls import path
from .views import HistoryListView, HistoryDeleteView, FavoriteListView

urlpatterns = [
    path('list/', HistoryListView.as_view(), name='history-list'),
    path('list/<int:pk>/', HistoryDeleteView.as_view(), name='history-delete'),
    path('favorites/', FavoriteListView.as_view(), name='favorites-list'),
]
