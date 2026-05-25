import os
import subprocess

# Define the Mermaid code for each diagram
diagrams = {
    'database_erd': """erDiagram
    USER {
        int id PK "Auto-incrementing user ID"
        string username "Unique username"
        string password "Hashed password"
        string email "User email address"
        string first_name "Optional first name"
        string last_name "Optional last name"
        boolean is_active "Active account flag"
        boolean is_staff "Admin access flag"
        boolean is_superuser "Superuser access flag"
        datetime last_login "Last authentication date"
        datetime date_joined "Account creation date"
    }

    TRANSLATION_RECORD {
        int id PK "Auto-incrementing translation ID"
        int user_id FK "Reference to USER"
        text english_text "The original English input text"
        text gujarati_text "The translated Gujarati output text"
        datetime created_at "Timestamp of translation"
    }

    GESTURE_HISTORY {
        int id PK "Auto-incrementing gesture log ID"
        int user_id FK "Reference to USER (Nullable for guest)"
        string detected_label "Predicted hand sign word/character"
        float confidence "ML model probability confidence score"
        string text_output "Final text representation"
        datetime created_at "Timestamp of detection"
    }

    FAVORITE_PHRASE {
        int id PK "Auto-incrementing favorite ID"
        int user_id FK "Reference to USER"
        string title "Short user-supplied name"
        string english_text "Original English phrase text"
        string gujarati_text "Translated Gujarati phrase text (Nullable)"
        datetime created_at "Timestamp when favorited"
    }

    USER ||--o{ TRANSLATION_RECORD : "has many"
    USER ||--o{ GESTURE_HISTORY : "has many"
    USER ||--o{ FAVORITE_PHRASE : "has many"
""",
    'system_architecture': """graph TD
    subgraph Client [Browser Client Application]
        UI["Web UI (HTML / CSS / JavaScript)"]
        Cam["Webcam Video Capture (mediaDevices)"]
        SR["SpeechRecognition Web API"]
    end

    subgraph Server [Django Backend Server]
        AS["ASGI Daphne Server (WebSockets)"]
        WS["WebSocket Consumer (GestureConsumer)"]
        API["Django View (REST APIs & Templates)"]
        
        subgraph MLService [ML / Gesture Processing Pipeline]
            GS["Gesture Service Coordinator"]
            PP["Image Preprocessing (OpenCV)"]
            MP["Hand Landmarks Extractor (MediaPipe)"]
            ML["Sign Classifier (Keras / CVZone Models)"]
        end

        subgraph TransService [Neural Translation Service]
            TS["Google Translator API (deep-translator)"]
        end
    end

    subgraph DB [Database Layer]
        SQLite[("SQLite Database (db.sqlite3)")]
    end

    UI -->|REST Requests| API
    UI -->|WebSocket Connection| AS
    AS --> WS
    Cam -->|Base64 Video Frames| WS
    SR -->|Speech-to-Text Input| UI

    WS -->|Forward Frame| GS
    API -->|Translate Request| TS
    
    GS --> PP
    PP -->|Decoded Frame| MP
    MP -->|63 Hand Landmarks| ML
    ML -->|Label Index & Confidence| GS
    GS -->|Translate Label| TS
    TS -->|Translation Output| GS

    API -->|Read / Write| SQLite
    WS -->|12. Save Detection Log| SQLite
""",
    'translation_sequence': """sequenceDiagram
    autonumber
    actor User as User
    participant Browser as Browser Client
    participant API as Django REST API
    participant TS as Translation Service
    participant DB as SQLite DB

    User->>Browser: Type English text or use Speech microphone
    Browser->>API: POST /api/translator/translate/ (Payload: {text})
    API->>TS: translate_to_gujarati(text)
    TS-->>API: Return Gujarati translation
    API-->>Browser: HTTP 200 OK (Response: {translated: "..."})
    Browser->>User: Render Gujarati text on UI
    
    User->>Browser: Click "Save to History"
    Browser->>API: POST /api/history/list/ (Payload: {english, gujarati, source})
    API->>DB: Save TranslationRecord entry
    DB-->>API: Confirm Save
    API-->>Browser: HTTP 201 Created
    Browser->>User: Show "Saved to history!" banner
""",
    'gesture_sequence': """sequenceDiagram
    autonumber
    actor User as User
    participant Browser as Browser Client
    participant WS as WebSocket Consumer (GestureConsumer)
    participant GS as Gesture Service Coordinator
    participant MP as MediaPipe Hands
    participant Model as Classifier Model (Keras)
    participant TS as Translation Service
    participant DB as SQLite DB

    User->>Browser: Access Gestures Page
    Browser->>WS: Establish WebSocket Handshake (ws://127.0.0.1:8000/ws/gesture/)
    WS-->>Browser: Welcome Connection Status: Live & Connected

    loop Every frame interval (e.g. 100ms)
        Browser->>Browser: Draw frame to hidden Canvas
        Browser->>WS: Send frame JSON (type: 'frame', image: base64, mode: 'words')
        WS->>GS: process_gesture_frame(base64, mode)
        GS->>GS: Decode base64 to OpenCV image matrix
        GS->>MP: Detect Hand Landmarks
        alt Case A: No hands detected
            MP-->>GS: Hand count: 0
            GS-->>WS: Return {"success": False, "status": "no_hands"}
            WS-->>Browser: Send socket JSON (type: 'prediction', success: False)
        else Case B: Hand detected
            MP-->>GS: Return 63 hand landmark coordinate array
            GS->>Model: Run model evaluation: getPrediction()
            Model-->>GS: Return predicted index & confidence score
            alt Confidence < Threshold
                GS-->>WS: Return {"success": False, "status": "low_confidence"}
                WS-->>Browser: Send socket JSON (type: 'prediction', success: False)
            else Confidence >= Threshold (e.g. >= 70%)
                GS->>TS: Translate predicted English word to Gujarati
                TS-->>GS: Return Gujarati word
                GS-->>WS: Return {"success": True, "label", "gujarati", "confidence"}
                WS->>DB: Save GestureHistory log entry (if logged in)
                DB-->>WS: Done
                WS-->>Browser: Send socket JSON (type: 'prediction', success: True, label, gujarati)
                Browser->>User: Display gesture word (English & Gujarati translation) on UI
            end
        end
    end
"""
}

# Directories
backend_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(backend_dir)
diagrams_dir = os.path.join(workspace_root, 'diagrams')
os.makedirs(diagrams_dir, exist_ok=True)

print(f"Generating PNG diagrams in: {diagrams_dir}")

for name, code in diagrams.items():
    mmd_path = os.path.join(diagrams_dir, f"{name}.mmd")
    png_path = os.path.join(diagrams_dir, f"{name}.png")
    
    # Write temporary mmd file
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(code)
        
    print(f"Rendering {name}.png...")
    # Execute mermaid-cli
    result = subprocess.run(
        f'npx @mermaid-js/mermaid-cli -i "{mmd_path}" -o "{png_path}" -b white',
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Successfully generated: {png_path}")
        # Clean up temporary mmd file
        if os.path.exists(mmd_path):
            os.remove(mmd_path)
    else:
        print(f"Error rendering {name}: {result.stderr}")

print("All PNG diagrams generated successfully!")
