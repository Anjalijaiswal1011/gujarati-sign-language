from django.urls import path
from . import consumers
import apps.gesture.routing

# Channels URLRouter does not support include(). 
# We combine the lists for the top-level router.
websocket_urlpatterns = [
    # A simple WebSocket endpoint for live application updates
    path('ws/live/', consumers.LiveUpdateConsumer.as_asgi()),
] + apps.gesture.routing.websocket_urlpatterns
