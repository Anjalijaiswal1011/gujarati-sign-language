from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .serializers import UserSerializer

# Models for data aggregation
from apps.history.models import TranslationRecord
from apps.gesture.models import GestureHistory
from apps.favorites.models import FavoritePhrase

class RegisterView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return Response({"message": "Login successful"}, status=status.HTTP_200_OK)
        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

class LogoutView(APIView):
    def post(self, request):
        logout(request)
        return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)

class ProfileView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)
        
        serializer = UserSerializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)

@login_required(login_url='/login/')
def dashboard_view(request):
    """
    Directly renders the premium dashboard with aggregated user statistics.
    """
    user = request.user
    
    # 1. Calculate Real-Time Counts
    translation_count = TranslationRecord.objects.filter(user=user).count()
    gesture_count = GestureHistory.objects.filter(user=user).count()
    favorites_count = FavoritePhrase.objects.filter(user=user).count()
    
    # 2. Fetch Recent Activities (Last 5)
    recent_translations = TranslationRecord.objects.filter(user=user).order_by('-created_at')[:5]
    recent_gestures = GestureHistory.objects.filter(user=user).order_by('-created_at')[:5]
    
    # 3. Combine for "Latest Activity" Feed
    # For a simple student project, we just pass lists directly
    context = {
        'translation_count': translation_count,
        'gesture_count': gesture_count,
        'favorites_count': favorites_count,
        'recent_translations': recent_translations,
        'recent_gestures': recent_gestures,
        'total_activity': translation_count + gesture_count
    }
    
    return render(request, 'dashboard.html', context)
