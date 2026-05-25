/**
 * script.js - Beginner-friendly JS logic for SignTranslate
 * 
 * Only handling the simple Mobile Navbar toggle interaction for now.
 * Kept clean and purely vanilla JS.
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // 1. Mobile Menu Toggle Setup
    const menuBtn = document.getElementById('mobile-menu-btn');
    const navMenu = document.getElementById('nav-menu');

    // Only run if the elements exist on the current page
    if (menuBtn && navMenu) {
        
        // Listen for user click on the hamburger button
        menuBtn.addEventListener('click', function() {
            
            // Toggle the 'active' CSS class on the nav menu (shows/hides it)
            navMenu.classList.toggle('active');
            
            // Update aria-expanded for screen readers (accessibility requirement)
            const isMenuOpen = navMenu.classList.contains('active');
            menuBtn.setAttribute('aria-expanded', isMenuOpen);
        });
    }
    
});
