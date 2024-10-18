document.addEventListener('DOMContentLoaded', () => {
    const typingBox = document.getElementById('typing-box');
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
        const speed = isDeleting || currentCharIndex > currentPhrase.length * 0.5 ? baseSpeed * 0.5 : baseSpeed; // Accélère à partir de 70% du texte

        if (isDeleting) {
            typingBox.textContent = currentPhrase.substring(0, currentCharIndex--);
            if (currentCharIndex < 0) {
                isDeleting = false;
                currentPhraseIndex = (currentPhraseIndex + 1) % phrases.length;
            }
        } else {
            typingBox.textContent = currentPhrase.substring(0, currentCharIndex++);
            if (currentCharIndex === currentPhrase.length) {
                setTimeout(() => isDeleting = true, 1000);  // Attend 1 seconde avant d'effacer
            }
        }
        setTimeout(type, speed);
    }

    type();
});
