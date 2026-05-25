import json
from channels.generic.websocket import AsyncWebsocketConsumer

class LiveUpdateConsumer(AsyncWebsocketConsumer):
    """
    A simple WebSocket Consumer that allows the server to push live 
    notifications and data (like recent history) out to connected browsers.
    We are not taking messages IN from the client here, just pushing OUT.
    """
    
    async def connect(self):
        # We assign everyone to a generic "global_updates" channel group.
        # In a more complex app, we might use f"user_{self.scope['user'].id}"
        self.group_name = "global_updates"

        # Join the channel group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        # Accept the WebSocket connection from the browser
        await self.accept()
        
        # Send a welcome message instantly
        await self.send(text_data=json.dumps({
            'type': 'status_update',
            'message': 'Connected to live server'
        }))

    async def disconnect(self, close_code):
        # Leave the group when the browser tab closes
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    # This method is triggered automatically when a Django View calls `group_send`
    async def app_notification(self, event):
        # The event dictionary has the data from the server.
        # Send the data down to the connected Javascript!
        await self.send(text_data=json.dumps({
            'type': 'history_update',
            'data': event['data']
        }))
