/**
 * Gesture Recognition Socket Logic
 * Core: Real-time WebSocket streaming & Speech Synthesis — PRESERVED EXACTLY
 * UI Layer: Updated only to support new DOM element IDs and class names
 * Scope: Gesture page only. No backend or API logic changes.
 */

// =====================================================
// DOM REFERENCES — All original IDs preserved
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    // =====================================================
    // DOM REFERENCES — All original IDs preserved
    // =====================================================
    const video       = document.getElementById('webcam');
    const canvas      = document.getElementById('canvas');
    const ctx         = canvas.getContext('2d', { alpha: false });
    const cameraBtn   = document.getElementById('camera-toggle');
    const streamBtn   = document.getElementById('stream-toggle');
    const statusBadge = document.getElementById('socket-status');
    const labelEl     = document.getElementById('prediction-label');
    const confText    = document.getElementById('confidence-text');
    const confFill    = document.getElementById('confidence-fill');
    const historyList = document.getElementById('history-list');
    const errorMsg    = document.getElementById('error-msg');
    const speakBtn    = document.getElementById('speak-btn');

    // Redesign elements
    const liveBadge       = document.getElementById('live-badge-el');
    const cameraStandby   = document.getElementById('camera-standby');
    const cameraActiveDot = document.getElementById('camera-active-dot');
    const gujaratiOutput  = document.getElementById('gujarati-output');
    const gujaratiEmpty   = document.getElementById('gujarati-empty');
    const gujaratiPanel   = document.getElementById('gujarati-panel');
    const historyEmpty    = document.getElementById('history-empty-state');
    const historyBadge    = document.getElementById('history-badge');
    const modeWordsBtn    = document.getElementById('mode-words');
    const modeAlphabetBtn = document.getElementById('mode-alphabet');

// =====================================================
// STATE — Unchanged from original logic
// =====================================================
let socket        = null;
let streamActive  = false;
let isStreaming   = false;
let lastSpoken    = "";
let streamInterval = null;
let audioEnabled  = true;
let historyCount  = 0;
let currentMode   = 'words';

const FRAME_INTERVAL = 200; // 5 FPS for smoother responsiveness

// =====================================================
// EVENT BINDINGS — Unchanged
// =====================================================
cameraBtn.addEventListener('click', toggleCamera);
streamBtn.addEventListener('click', toggleStreaming);

// Mode switching
modeWordsBtn.addEventListener('click', () => setMode('words'));
modeAlphabetBtn.addEventListener('click', () => setMode('alphabet'));

function setMode(mode) {
    currentMode = mode;
    if (mode === 'words') {
        modeWordsBtn.classList.add('active', 'g-btn-primary');
        modeWordsBtn.classList.remove('g-btn-outline');
        modeAlphabetBtn.classList.remove('active', 'g-btn-primary');
        modeAlphabetBtn.classList.add('g-btn-outline');
    } else {
        modeAlphabetBtn.classList.add('active', 'g-btn-primary');
        modeAlphabetBtn.classList.remove('g-btn-outline');
        modeWordsBtn.classList.remove('active', 'g-btn-primary');
        modeWordsBtn.classList.add('g-btn-outline');
    }
}

// Initial mode UI
setMode('words');

// Audio toggle — UI updated, function logic unchanged
speakBtn.addEventListener('click', () => {
    audioEnabled = !audioEnabled;
    if (audioEnabled) {
        speakBtn.classList.replace('g-btn-toggle-off', 'g-btn-toggle-on');
        speakBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i><span>On</span>';
    } else {
        speakBtn.classList.replace('g-btn-toggle-on', 'g-btn-toggle-off');
        speakBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i><span>Off</span>';
    }
});

// =====================================================
// CAMERA TOGGLE — Core logic unchanged
// UI: shows/hides standby overlay and live badge
// =====================================================
async function toggleCamera() {
    if (!streamActive) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            video.srcObject = stream;
            streamActive = true;

            // Update button UI
            cameraBtn.innerHTML = '<i class="fa-solid fa-camera-slash"></i><span>Stop Camera</span>';
            cameraBtn.classList.replace('g-btn-outline', 'g-btn-danger');

            // Show camera active state
            if (cameraStandby)   cameraStandby.classList.add('hidden');
            if (liveBadge)       liveBadge.classList.add('visible');
            if (cameraActiveDot) cameraActiveDot.classList.add('active');

            streamBtn.disabled = false;
            initWebSocket();
        } catch (err) {
            errorMsg.textContent = 'Camera error: ' + err.message;
        }
    } else {
        // Stop tracks
        const tracks = video.srcObject?.getTracks() || [];
        tracks.forEach(t => t.stop());
        video.srcObject = null;
        streamActive = false;

        // Reset button UI
        cameraBtn.innerHTML = '<i class="fa-solid fa-camera"></i><span>Start Camera</span>';
        cameraBtn.classList.replace('g-btn-danger', 'g-btn-outline');

        // Restore standby state
        if (cameraStandby)   cameraStandby.classList.remove('hidden');
        if (liveBadge)       liveBadge.classList.remove('visible');
        if (cameraActiveDot) cameraActiveDot.classList.remove('active');

        streamBtn.disabled = true;
        stopStreaming();
    }
}

// =====================================================
// WEBSOCKET INIT — Core logic unchanged
// UI: updates status badge text and classes
// =====================================================
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const port = window.location.port ? `:${window.location.port}` : '';
    const wsUrl = `${protocol}//${window.location.hostname}${port}/ws/gesture/stream/`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        statusBadge.classList.replace('status-disconnected', 'status-connected');
        statusBadge.innerHTML = '<span class="status-dot"></span><span class="status-text">Connected</span>';
    };

    socket.onclose = () => {
        statusBadge.classList.replace('status-connected', 'status-disconnected');
        statusBadge.innerHTML = '<span class="status-dot"></span><span class="status-text">Disconnected</span>';
        stopStreaming();
    };

    socket.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'prediction') {
            updateUI(data);
        }
    };
}

// =====================================================
// STREAMING TOGGLE — Core logic unchanged
// UI: button switches to danger/primary style
// =====================================================
function toggleStreaming() {
    if (!isStreaming) {
        isStreaming = true;
        streamBtn.innerHTML = '<i class="fa-solid fa-stop"></i><span>Stop Detection</span>';
        streamBtn.classList.replace('g-btn-primary', 'g-btn-danger');
        streamInterval = setInterval(sendFrame, FRAME_INTERVAL);
    } else {
        stopStreaming();
    }
}

function stopStreaming() {
    isStreaming = false;
    streamBtn.innerHTML = '<i class="fa-solid fa-play"></i><span>Start Detection</span>';
    if (streamBtn.classList.contains('g-btn-danger')) {
        streamBtn.classList.replace('g-btn-danger', 'g-btn-primary');
    }
    clearInterval(streamInterval);

    // Reset prediction card state
    const predCard = document.getElementById('prediction-card');
    if (predCard) predCard.classList.remove('active');
}

// =====================================================
// SEND FRAME — Core logic unchanged
// =====================================================
function sendFrame() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    canvas.width  = 300;
    canvas.height = 225;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frame = canvas.toDataURL('image/jpeg', 0.3); // Lower quality for speed
    socket.send(JSON.stringify({
        type:  'frame',
        image: frame,
        mode:  currentMode
    }));
}

// =====================================================
// UPDATE UI — Core logic unchanged
// UI: adds Gujarati output panel update
// =====================================================
function updateUI(data) {
    const predictionCard = document.getElementById('prediction-card');
    const modelWarning   = document.getElementById('model-warning');

    // Show/Hide simulation warning
    if (modelWarning) {
        modelWarning.style.display = data.is_mock ? 'block' : 'none';
    }

    if (data.success) {
        // Update English prediction label
        labelEl.textContent = data.label;

        // Update confidence bar
        const pct = data.confidence.toFixed(0);
        confText.textContent = `${pct}%`;
        confFill.style.width = `${data.confidence}%`;

        // Activate the prediction card styling
        if (predictionCard) predictionCard.classList.add('active');

        // Update Gujarati output panel (visual layer only)
        updateGujaratiPanel(data.gujarati || data.label);

        addToHistory(data.label, data.gujarati || null);
        speakGesture(data.label, data.gujarati || null);
    } else {
        // Handle "Hand Detected" feedback even in Mock/Idle mode
        if (data.hand_count > 0) {
            labelEl.textContent = 'Hand Detected';
            labelEl.style.fontSize = '1.2rem';
            labelEl.style.opacity = '0.6';
            if (predictionCard) predictionCard.classList.add('active');
        } else {
            labelEl.textContent = '---';
            labelEl.style.fontSize = '';
            labelEl.style.opacity = '';
            if (predictionCard) predictionCard.classList.remove('active');
        }

        confText.textContent = '0%';
        confFill.style.width = '0%';

        // Reset Gujarati panel
        clearGujaratiPanel();
    }
}

// =====================================================
// GUJARATI PANEL HELPER — UI visual layer only
// Does NOT change any backend or processing logic
// =====================================================
function updateGujaratiPanel(gujaratiText) {
    if (gujaratiOutput && gujaratiEmpty && gujaratiPanel) {
        gujaratiOutput.textContent = gujaratiText;
        gujaratiOutput.classList.add('visible');
        gujaratiEmpty.classList.add('hidden');
        gujaratiPanel.classList.add('has-translation');
    }
}

function clearGujaratiPanel() {
    if (gujaratiOutput && gujaratiEmpty && gujaratiPanel) {
        gujaratiOutput.textContent = '';
        gujaratiOutput.classList.remove('visible');
        gujaratiEmpty.classList.remove('hidden');
        gujaratiPanel.classList.remove('has-translation');
    }
}

// =====================================================
// ADD TO HISTORY — Core logic unchanged
// UI: adds Gujarati chip to each history item
// =====================================================
function addToHistory(label, gujaratiLabel) {
    if (historyEmpty) historyEmpty.classList.add('hidden');

    historyCount = Math.min(historyCount + 1, 5);
    if (historyBadge) historyBadge.textContent = String(historyCount);

    const li = document.createElement('li');
    li.className = 'history-item';

    const now = new Date().toLocaleTimeString([], {
        hour:   '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    const gjChip = gujaratiLabel
        ? `<span class="history-gj">${gujaratiLabel}</span>`
        : '';

    li.innerHTML = `
        <div class="history-sign">
            <i class="fa-solid fa-hand-back-fist"></i>
            <span class="history-en">${label}</span>
            ${gjChip}
        </div>
        <span class="history-time">${now}</span>
    `;

    historyList.prepend(li);
    if (historyList.children.length > 5) historyList.lastElementChild.remove();
}

// =====================================================
// SPEAK GESTURE — Now speaks Gujarati!
// Respects audioEnabled toggle
// =====================================================
function speakGesture(label, gujaratiText) {
    if (!audioEnabled) return;
    if (label !== lastSpoken && 'speechSynthesis' in window) {
        lastSpoken = label;
        
        const textToSpeak = gujaratiText || label;
        const msg = new SpeechSynthesisUtterance(textToSpeak);
        
        // Set language to Gujarati if available, else fallback to English
        msg.lang = gujaratiText ? 'gu-IN' : 'en-US';
        
        window.speechSynthesis.speak(msg);
        setTimeout(() => { lastSpoken = ''; }, 3000);
    }
}

});
