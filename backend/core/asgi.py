import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Ensure Django forms its initial configuration before importing routes
django_asgi_app = get_asgi_application()

import core.routing

application = ProtocolTypeRouter({
    # standard Django HTTP routing
    "http": django_asgi_app,
    
    # WebSocket routing using Auth middleware so we know who is connected (request.user)
    "websocket": AuthMiddlewareStack(
        URLRouter(
            core.routing.websocket_urlpatterns
        )
    ),
})
