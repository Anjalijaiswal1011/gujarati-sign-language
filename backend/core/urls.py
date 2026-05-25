from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LogoutView
from apps.users import views as views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Frontend Pages
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('login/', TemplateView.as_view(template_name='login.html'), name='login-page'),
    path('register/', TemplateView.as_view(template_name='register.html'), name='register-page'),
    path('logout/', LogoutView.as_view(next_page='login-page'), name='logout'),
    path('camera-test/', TemplateView.as_view(template_name='camera_test.html'), name='camera-test'),
    
    # Protected pages
    path('dashboard/', include([
        path('', views.dashboard_view, name='dashboard'),
    ])),
    path('translation/', login_required(TemplateView.as_view(template_name='translation.html'), login_url='/login/'), name='translation'),
    path('gesture/', include('apps.gesture.urls')),
    path('history/', login_required(TemplateView.as_view(template_name='history.html'), login_url='/login/'), name='history'),
    path('favorites/', login_required(TemplateView.as_view(template_name='favorites.html'), login_url='/login/'), name='favorites'),

    # API endpoints
    path('api/users/', include('apps.users.urls')),
    path('api/history/', include('apps.history.urls')),
    path('api/translator/', include('apps.translator.urls')),
    path('api/gesture/', include('apps.gesture.urls')),
]

from django.conf import settings
from django.views.static import serve
from django.urls import re_path

urlpatterns += [
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]

