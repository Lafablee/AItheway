console.log('script.js loading...')
// Error handling function with enhanced AJAX support
function showErrorOverlay(title, message, isAuthError = false) {
    console.log('showErrorOverlay called with:', { title, message, isAuthError });
    // Add keyframe animation for spinner and scaling
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap');
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        .error-button {
            font-family: 'Montserrat', sans-serif;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            min-width: 160px;
            transform: translateY(0);
            transition: all 0.2s ease;
        }
        .error-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0, 123, 255, 0.25);
            background-color: #0056b3;
        }
    `;
    document.head.appendChild(styleSheet);

    // Helper function to get icon color based on error type
    const getIconColor = (isAuth) => {
        return isAuth ? '#FFA500' : '#dc3545';
    };

    // Remove existing overlay if any
    let overlay = document.getElementById('error-overlay');
    if (overlay) {
        overlay.remove();
    }

    overlay = document.createElement('div');
    overlay.id = 'error-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        backdrop-filter: blur(2px);
        padding: 20px;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;

    const buttonText = isAuthError ? 'Retour Ã  la connexion' : 'RÃ©essayer';
    const buttonAction = isAuthError ?
        `window.location.href='${LOGIN_URL}'` :
        'window.location.reload()';

    const spinner = `
        <div style="
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0, 123, 255, 0.2);
            border-top: 2px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 8px;
            vertical-align: middle;
        "></div>
    `;

    const redirectMessage = isAuthError ? `
        <div style="
            margin-top: 15px;
            font-size: 13px;
            color: #666;
            font-style: italic;
        ">
            ${spinner}
            Redirection automatique dans <span id="countdown">3</span> secondes...
        </div>
    ` : '';

    overlay.innerHTML = `
        <div style="
            font-family: 'Montserrat', sans-serif;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 380px;
            margin: 0 auto;
            text-align: center;
            position: relative;
            top: -10%;
            animation: scaleIn 0.3s ease forwards;
        ">
            <div style="
                margin: 0 auto 15px;
                padding: 12px;
                background: ${getIconColor(isAuthError)}1A;
                border-radius: 50%;
                width: fit-content;
            ">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" style="color: ${getIconColor(isAuthError)}">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
            </div>
            <h3 style="font-family: 'Montserrat', sans-serif; color: #333; margin: 0 0 15px 0; font-size: 18px; font-weight: 600;">${title}</h3>
            <p style="font-family: 'Montserrat', sans-serif; color: #666; margin: 0 0 20px 0; font-size: 14px; line-height: 1.5;">${message}</p>
            <button class="error-button" onclick="${buttonAction}">${buttonText}</button>
            ${redirectMessage}
        </div>
    `;

    document.body.appendChild(overlay);

    // Add click outside to close for non-auth errors
    if (!isAuthError) {
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                overlay.style.opacity = '0';
                setTimeout(() => overlay.remove(), 300);
            }
        });
    }

    // Trigger fade in
    requestAnimationFrame(() => {
        overlay.style.opacity = '1';
    });

    // Add countdown and redirect for auth errors
    if (isAuthError) {
        let countdown = 3;
        const countdownElement = document.getElementById('countdown');
        const timer = setInterval(() => {
            countdown--;
            if (countdownElement) {
                countdownElement.textContent = countdown;
            }
            if (countdown <= 0) {
                clearInterval(timer);
                window.location.href = LOGIN_URL;
            }
        }, 1000);
    }
}

// Helper function to handle API responses
async function handleApiResponse(response) {
    const result = await response.json();

    if (response.status === 401) {
        showErrorOverlay(
            'Session ExpirÃ©e',
            'Votre session a expirÃ©. Veuillez vous reconnecter.',
            true
        );
        return null;
    }

    if (response.status === 403) {
        showErrorOverlay(
            'Erreur d\'authentification',
            'AccÃ¨s non autorisÃ©. Veuillez vous reconnecter.',
            true
        );
        return null;
    }

    if (!response.ok) {
        showErrorOverlay('Erreur', result.error || 'Une erreur est survenue');
        return null;
    }

    return result;
}

// Keep your existing typing animation code
document.addEventListener('DOMContentLoaded', () => {
    const typingBox = document.getElementById('typing-box');
    if (typingBox) {  // Only run if typing-box exists
        const phrases = [
            "Coloriser les images en noir et blanc",
            "GÃ©nÃ©rer des images uniques avec IA",
            "AmÃ©liorer la qualitÃ© de vos photos",
            "Supprimer l'arriÃ¨re-plan des images",
            "Vectoriser des logos et des dessins"
        ];
        let currentPhraseIndex = 0;
        let currentCharIndex = 0;
        let isDeleting = false;

        function type() {
            const currentPhrase = phrases[currentPhraseIndex];
            const baseSpeed = isDeleting ? 50 : 150;
            const speed = isDeleting || currentCharIndex > currentPhrase.length * 0.5 ? baseSpeed * 0.5 : baseSpeed;

            if (isDeleting) {
                typingBox.textContent = currentPhrase.substring(0, currentCharIndex--);
                if (currentCharIndex < 0) {
                    isDeleting = false;
                    currentPhraseIndex = (currentPhraseIndex + 1) % phrases.length;
                }
            } else {
                typingBox.textContent = currentPhrase.substring(0, currentCharIndex++);
                if (currentCharIndex === currentPhrase.length) {
                    setTimeout(() => isDeleting = true, 1000);
                }
            }
            setTimeout(type, speed);
        }

        type();
    }
});

// Export the functions for global use
window.showErrorOverlay = showErrorOverlay;
console.log('script.js functions exported to window');
window.handleApiResponse = handleApiResponse;