// Utility function to show toast messages
function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.style.backgroundColor = isError ? "#D32F2F" : "#4CAF50";
    toast.classList.remove("hidden");
    
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3000);
}

// Utility to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Global fetch wrapper with credentials
async function apiFetch(url, options = {}) {
    options.credentials = 'same-origin';
    if (!options.headers) {
        options.headers = {};
    }
    options.headers['X-CSRFToken'] = getCookie('csrftoken');
    options.headers['Content-Type'] = 'application/json';
    
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        return { response, data };
    } catch (error) {
        return { response: { ok: false }, data: { error: error.message } };
    }
}

// Navigation Logic Based on Login Status
async function checkAuthStatus() {
    const { response } = await apiFetch('/api/users/profile/');
    
    const loginLink = document.getElementById('nav-login');
    const registerLink = document.getElementById('nav-register');
    const dashboardLink = document.getElementById('nav-dashboard');
    const logoutLink = document.getElementById('nav-logout');

    if (response.ok) {
        // User is logged in
        if (loginLink) loginLink.classList.add('hidden');
        if (registerLink) registerLink.classList.add('hidden');
        if (dashboardLink) dashboardLink.classList.remove('hidden');
        if (logoutLink) logoutLink.classList.remove('hidden');
    }
}

// Common Logout functionality
document.addEventListener("DOMContentLoaded", () => {
    checkAuthStatus();

    // Logout is handled by the standard Django form in base.html

    // --- WEBSOCKET INTEGRATION ---
    // Establish a live connection to Django Channels
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = protocol + window.location.host + '/ws/live/';
    
    // Only connect if user is likely logged in (avoiding errors on login page)
    if (!window.location.pathname.includes('login') && !window.location.pathname.includes('register')) {
        const liveSocket = new WebSocket(wsUrl);

        liveSocket.onopen = function(e) {
            console.log("WebSocket connected globally.");
        };

        liveSocket.onmessage = function(e) {
            const payload = JSON.parse(e.data);
            
            // Handle different types of live messages from backend
            if (payload.type === 'status_update') {
                console.log(payload.message); // e.g. "Connected to live server"
            } 
            else if (payload.type === 'history_update') {
                // Flash a global notification
                showToast(`Live Activity: New ${payload.data.source} translation saved!`);
                
                // If we are currently on the History page, auto-refresh the grid!
                if (typeof window.loadHistory === 'function') {
                    window.loadHistory();
                } else {
                    const historyContainer = document.getElementById('history-container');
                    if (historyContainer) {
                        const newHtml = `
                            <div class="card" style="box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; padding: 15px; border-left: 4px solid #2196F3; animation: fadeIn 1s;">
                                <div style="display: flex; justify-content: space-between;">
                                    <strong>${payload.data.source} Input (Live)</strong>
                                    <button class="btn btn-danger" style="padding: 5px 10px; font-size: 0.8rem;" disabled>Just Added</button>
                                </div>
                                <p style="margin-top: 10px;"><strong>English:</strong> ${payload.data.english_text}</p>
                                <p style="font-size: 1.2rem; margin-top: 5px;"><strong>Gujarati:</strong> ${payload.data.gujarati_text}</p>
                            </div>
                        `;
                        // Prepend instantly without needing an API refresh
                        historyContainer.insertAdjacentHTML('afterbegin', newHtml);
                    }
                }
            }
        };

        liveSocket.onerror = function(e) {
            console.error("Live WebSockets error", e);
        };
        
        liveSocket.onclose = function(e) {
            console.log("Live WebSockets gracefully closed.");
        };
    }
});
