import os
import time
import django
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from playwright.sync_api import sync_playwright

# Setup Django Environment to access models and settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.contrib.auth.models import User
from apps.history.models import TranslationRecord
from apps.gesture.models import GestureHistory
from apps.favorites.models import FavoritePhrase

def setup_dummy_data():
    """
    Ensures a test user exists and populates some initial history, gesture records,
    and favorites so the screenshots show actual rich data rather than blank screens.
    """
    print("Setting up dummy user and mock data...")
    username = 'screenshot_test'
    password = 'testpassword'
    
    # 1. Create or fetch user
    user, created = User.objects.get_or_create(username=username, defaults={'email': 'test@example.com'})
    if created or not user.check_password(password):
        user.set_password(password)
        user.save()
        print(f"User '{username}' created/updated.")
    
    # 2. Add Translation Record history if empty
    if not TranslationRecord.objects.filter(user=user).exists():
        TranslationRecord.objects.create(
            user=user,
            english_text="Hello, how are you?",
            gujarati_text="નમસ્તે, તમે કેમ છો?"
        )
        TranslationRecord.objects.create(
            user=user,
            english_text="Please help me find the way.",
            gujarati_text="કૃપા કરીને મને રસ્તો શોધવામાં મદદ કરો."
        )
        TranslationRecord.objects.create(
            user=user,
            english_text="What is your name?",
            gujarati_text="તમારું નામ શું છે?"
        )
        print("Created mock translation history records.")

    # 3. Add Gesture History if empty
    if not GestureHistory.objects.filter(user=user).exists():
        GestureHistory.objects.create(
            user=user,
            detected_label="Hello",
            confidence=0.96,
            text_output="Hello"
        )
        GestureHistory.objects.create(
            user=user,
            detected_label="Yes",
            confidence=0.98,
            text_output="Yes"
        )
        GestureHistory.objects.create(
            user=user,
            detected_label="Help",
            confidence=0.92,
            text_output="Help"
        )
        print("Created mock gesture detection history records.")

    # 4. Add Favorites if empty
    if not FavoritePhrase.objects.filter(user=user).exists():
        FavoritePhrase.objects.create(
            user=user,
            title="Greeting",
            english_text="Hello, how are you?",
            gujarati_text="નમસ્તે, તમે કેમ છો?"
        )
        FavoritePhrase.objects.create(
            user=user,
            title="Emergency",
            english_text="I need assistance.",
            gujarati_text="મને મદદની જરૂર છે."
        )
        print("Created mock favorite records.")

    print("Database preparation done.")
    return username, password

def take_screenshots():
    # Setup dummy data first
    username, password = setup_dummy_data()
    
    # Locate/create screenshots directory in workspace root
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(backend_dir)
    screenshot_dir = os.path.join(workspace_root, 'screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"Screenshots directory: {screenshot_dir}")

    base_url = 'http://127.0.0.1:8000'

    # Define the pages we want to screenshot
    # Format: { filename_prefix: (url_path, requires_login) }
    pages = {
        '1_home': ('/', False),
        '2_login': ('/login/', False),
        '3_register': ('/register/', False),
        '4_camera_test': ('/camera-test/', False),
        '5_dashboard': ('/dashboard/', True),
        '6_translation': ('/translation/', True),
        '7_gesture': ('/gesture/', True),
        '8_history': ('/history/', True),
        '9_favorites': ('/favorites/', True),
    }

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with sync_playwright() as p:
        print("Launching Playwright Chromium...")
        browser = p.chromium.launch(headless=True)
        # Use high resolution desktop viewport with double device scale (Retina clarity)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=2
        )
        page = context.new_page()

        # Step 1: Capture all public pages
        for name, (path, req_login) in pages.items():
            if not req_login:
                url = f"{base_url}{path}"
                print(f"Loading public page: {url}")
                page.goto(url)
                # Wait for any minor styles, icons, or transitions to finish loading
                page.wait_for_timeout(1000)
                screenshot_path = os.path.join(screenshot_dir, f"{name}.png")
                page.screenshot(path=screenshot_path, full_page=True)
                print(f"Saved: {screenshot_path}")

        # Step 2: Navigate to login and authenticate
        login_url = f"{base_url}/login/"
        print(f"Authenticating via: {login_url}")
        page.goto(login_url)
        page.fill('#username', username)
        page.fill('#password', password)
        page.click('button[type="submit"]')
        
        # Wait for the dashboard page redirect
        page.wait_for_url(f"{base_url}/dashboard/")
        print("Successfully logged in.")
        page.wait_for_timeout(1000)

        # Step 3: Capture all authenticated pages
        for name, (path, req_login) in pages.items():
            if req_login:
                url = f"{base_url}{path}"
                print(f"Loading authenticated page: {url}")
                page.goto(url)
                # Let API calls complete and fill the containers (e.g. history & favorites)
                if 'history' in name or 'favorites' in name:
                    page.wait_for_timeout(1500)
                else:
                    page.wait_for_timeout(1000)
                
                screenshot_path = os.path.join(screenshot_dir, f"{name}.png")
                page.screenshot(path=screenshot_path, full_page=True)
                print(f"Saved: {screenshot_path}")

        browser.close()
        print("Screen capture completed!")

if __name__ == '__main__':
    take_screenshots()
