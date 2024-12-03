// Error handling function
function showErrorOverlay(title, message) {
    let overlay = document.getElementById('error-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'error-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        `;
        
        overlay.innerHTML = `
            <div style="
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                width: 90%;
                text-align: center;
            ">
                <h3 style="color: #333; margin-bottom: 15px;">${title}</h3>
                <p style="color: #666; margin-bottom: 20px;">${message}</p>
                <button onclick="window.location.reload()" style="
                    background-color: #007bff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                ">Réessayer</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }
}

// Typing animation
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

// Export the error handling function
window.showErrorOverlay = showErrorOverlay;