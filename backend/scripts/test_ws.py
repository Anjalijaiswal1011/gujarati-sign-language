import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/gesture/stream/"
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as ws:
            print("Connected! Sending a ping...")
            # Send an empty frame to see response
            await ws.send(json.dumps({"type": "frame", "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==", "mode": "words"}))
            response = await ws.recv()
            print("Response:", response)
    except Exception as e:
        print("Connection failed:", e)

if __name__ == "__main__":
    asyncio.run(test_ws())
