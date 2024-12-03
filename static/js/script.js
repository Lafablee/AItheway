// Error handling function with enhanced AJAX support
function showErrorOverlay(title, message, isAuthError = false) {
    // Add keyframe animation for spinner
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
         from { opacity: 0; }
         to { opacity: 1; }
        }
    `;
    document.head.appendChild(styleSheet);

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
        transition: oppacity 0.3s ease;
    `;

    const buttonText = isAuthError ? 'Retour à la connexion' : 'Réessayer';
    const buttonAction = isAuthError ?
        `window.location.href='${LOGIN_URL}'` :
        'window.location.reload()';

    const spinner = `
       <div style="
         width: 16px;
         height: 16px;
         border: 2px solid #f3f3f3;
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
            font-style: oblique;
        ">
          ${spinner}
          Redirection automatique dans <span id="countdown">3</span> secondes...
        </div>
    `:'';

    overlay.innerHTML = `
        <div style="
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);
            max-width: 380px;
            width: 100%;
            margin: 0 auto;
            text-align: center;
            position: relative;
            top: -10%;
            animation: fadeIn 0.3s ease forwards;
        ">
            <h3 style="color: #333; margin: 0 0 15px 0; font-size: 18px; font-weight: 600; ">${title}</h3>
            <p style="color: #666; margin: 0 0 20px 0; font-size: 14px; line-height: 1.5;">${message}</p>
            <button onclick="${buttonAction}" style="
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                transition: background-color 0.3s ease;
                font-size: 14px;
                width: auto;
                min-width: 160px;
            ">${buttonText}</button>
            ${redirectMessage}
        </div>
    `;

    document.body.appendChild(overlay);

     // Add click outside to close for non-auth errors
    if (!isAuthError) {
        overlay.addEventListener('click', function(e) {
             if (e.target === overlay) {
                 overlay.style.opacity = '0';
                 setTimeout(() =>overlay.remove(), 300); // Remove after fade out
             }
        });
    }
    // Trigger fade in
    requestAnimationFrame(() =>{
        overlay.style.opacity = '1';
    });

    // Auto redirect for auth errors after 3 seconds
    if (isAuthError) {
        let countdown = 3;
        const countdownElement = document.getElementById('countdown');
        const timer = setInterval(() => {
            countdown--;
            if (countdownElement){
                countdownElement.textContent = countdown;
            }
            if (countdown <= 0){
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
            'Session Expirée',
            'Votre session a expiré. Veuillez vous reconnecter.',
            true
        );
        return null;
    }

    if (response.status === 403) {
        showErrorOverlay(
            'Erreur d\'authentification',
            'Accès non autorisé. Veuillez vous reconnecter.',
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
            "Générer des images uniques avec IA",
            "Améliorer la qualité de vos photos",
            "Supprimer l'arrière-plan des images",
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
window.handleApiResponse = handleApiResponse;