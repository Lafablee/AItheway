console.log('script.js loading...')
// Error handling function with enhanced AJAX support

class ErrorHandler {
    static async showError(title, message, options = {}) {
        // Configurer les options par défaut
        const defaultOptions = {
            isAuthError: false,
            autoHide: true,
            duration: 5000
        };
        options = {...defaultOptions, ...options};

        // Créer et configurer l'overlay
        const overlay = document.createElement('div');
        overlay.className = 'error-overlay';
        overlay.setAttribute('role', 'alert');
        overlay.setAttribute('aria-live', 'assertive');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            backdrop-filter: blur(4px);
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        // Créer le contenu de l'erreur
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            max-width: 400px;
            width: 90%;
            text-align: center;
            transform: translateY(-20px);
            transition: transform 0.3s ease;
        `;

        // Ajouter le titre
        const titleElement = document.createElement('h4');
        titleElement.textContent = title;
        titleElement.style.cssText = `
            color: ${options.isAuthError ? '#FFA500' : '#dc3545'};
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
        `;

        // Ajouter le message
        const messageElement = document.createElement('p');
        messageElement.textContent = message;
        messageElement.style.cssText = `
            color: #666;
            margin: 0 0 20px 0;
            font-size: 14px;
            line-height: 1.5;
        `;
        // Ajouter le bouton
        const button = document.createElement('button');
        button.textContent = options.isAuthError ? 'Retour à la connexion' : 'Fermer';
        button.style.cssText = `
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        `;
        // Ajouter les éléments au DOM
        content.appendChild(titleElement);
        content.appendChild(messageElement);
        content.appendChild(button);
        overlay.appendChild(content);
        document.body.appendChild(overlay);

        // Animation d'entrée
        requestAnimationFrame(() => {
            overlay.style.opacity = '1';
            content.style.transform = 'translateY(0)';
        });

        // Gérer les événements
        button.onclick = () => {
            if (options.isAuthError) {
                window.location.href = LOGIN_URL;
            } else {
                this.closeError(overlay);
            }
        };

        //Fermeture au clic en dehors
        if (!options.isAuthError) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.closeError(overlay);
                }
            });
        }

        // Auto-hide si configuré
        if (options.autoHide && !options.isAuthError) {
            setTimeout(() => this.closeError(overlay), options.duration);
        }

        // Redirection automatique pour les erreurs d'auth
        if (options.isAuthError) {
            setTimeout(() => {
                window.location.href = LOGIN_URL;
            }, 3000);
        }
    }
    static closeError(overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 300);
    }
    static async handleApiError(response) {
        try {
            const errorData = await response.json();
            this.showError(
                errorData.title || 'Erreur',
                errorData.error,
                {
                    isAuthError: response.status === 401 || response.status === 403,
                    autoHide: response.status !== 401
                }
            );
        } catch (e) {
            this.showError('Erreur', 'Une erreur inattendue est survenue');
        }
    }
}


// Helper function to handle API responses
async function handleApiResponse(response) {

    if (!response.ok) {
        await ErrorHandler.handleApiError(response);
        return null;
    }
    return response.json();
}
// Exporter les fonctions pour une utilisation globale
window.ErrorHandler = ErrorHandler;
window.handleApiResponse = handleApiResponse;

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
console.log('script.js functions exported to window');
window.handleApiResponse = handleApiResponse;