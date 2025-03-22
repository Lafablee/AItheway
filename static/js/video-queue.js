// Système de file d'attente simulée pour la génération vidéo
class VideoQueueSimulator {
  constructor(options = {}) {
    // Configuration par défaut
    this.config = {
      containerId: 'video-generation-status', // ID du conteneur HTML
      taskId: null,                           // ID de la tâche MiniMax
      isPremium: false,                       // Compte premium ou gratuit
      checkInterval: 3000,                    // Intervalle de vérification MiniMax (ms)
      statusEndpoint: '/check_video_status/',  // Endpoint pour vérifier le statut

      // Simulation de file d'attente
      initialQueuePositionMin: 3,             // Position minimale initiale (premium)
      initialQueuePositionMax: 8,             // Position maximale initiale (premium)
      freeUserMultiplier: 2,                  // Multiplicateur pour comptes gratuits
      queueUpdateIntervalMin: 2000,           // Intervalle min entre mises à jour (ms)
      queueUpdateIntervalMax: 5000,           // Intervalle max entre mises à jour (ms)

      // Messages et animations
      loadingMessages: {
        queue: [
          "Préparation de votre vidéo...",
          "Chargement des ressources...",
          "Optimisation du prompt...",
          "Initialisation des paramètres vidéo..."
        ],
        processing: [
          "Analyse de votre demande...",
          "Génération de la séquence...",
          "Rendu des images...",
          "Traitement des frames vidéo...",
          "Composition des éléments visuels...",
          "Application des effets spéciaux..."
        ],
        finalizing: [
          "Assemblage de la vidéo...",
          "Optimisation du résultat...",
          "Finalisation du rendu...",
          "Préparation du téléchargement...",
          "Votre vidéo apparaîtra d'une seconde à l'autre..."
        ]
      },

      // Messages spécifiques aux abonnements
      subscriptionMessages: {
        free: "⚠️ Les comptes gratuits peuvent connaître des délais plus longs. Passez à un abonnement premium pour une génération prioritaire!",
        premium: "✨ Génération prioritaire activée grâce à votre abonnement premium!"
      }
    };

    // Fusion avec les options fournies
    Object.assign(this.config, options);

    // État interne
    this.state = {
      queuePosition: 0,
      phase: 'queue',  // 'queue', 'processing', 'finalizing', 'completed'
      realStatus: 'Queueing',
      startTime: Date.now(),
      videoUrl: null,
      messageIndex: 0,
      errorMessage: null
    };

    // Initialisation
    this.init();
  }

  init() {
    // Obtenir le conteneur
    this.container = document.getElementById(this.config.containerId);
    if (!this.container) {
      console.error("Conteneur de statut vidéo non trouvé!");
      return;
    }

    // Créer les éléments HTML
    this.createStatusElements();

    // Déterminer la position initiale dans la file (plus petite pour les comptes premium)
    const multiplier = this.config.isPremium ? 1 : this.config.freeUserMultiplier;
    const min = this.config.initialQueuePositionMin * multiplier;
    const max = this.config.initialQueuePositionMax * multiplier;
    this.state.queuePosition = Math.floor(Math.random() * (max - min + 1)) + min;

    // Commencer la simulation
    this.startQueueSimulation();

    // Commencer la vérification réelle du statut
    this.startStatusChecking();
  }

  createStatusElements() {
    this.container.innerHTML = `
      <div class="video-generation-wrapper">
        <div class="status-header">
          <h3>Génération de vidéo en cours</h3>
          <div class="subscription-badge">${this.config.isPremium ? this.config.subscriptionMessages.premium : this.config.subscriptionMessages.free}</div>
        </div>
        
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
          <div class="progress-text">Initialisation...</div>
        </div>
        
        <div class="status-details">
          <div class="queue-position-container">
            <div class="queue-label">Position dans la file d'attente:</div>
            <div class="queue-value">${this.state.queuePosition}</div>
          </div>
          
          <div class="status-message-container">
            <div class="status-spinner"></div>
            <div class="status-message">Démarrage de la génération...</div>
          </div>
          
          <div class="minimax-status">Statut: <span class="status-value">En attente</span></div>
          
          <div class="elapsed-time">Temps écoulé: <span class="time-value">0:00</span></div>
        </div>
        
        <div class="error-container" style="display: none;"></div>
        <div class="video-result-container" style="display: none;">
          <video controls class="result-video" style="max-width: 100%;">
            Votre navigateur ne prend pas en charge la lecture vidéo.
          </video>
          <div class="video-actions">
            <button class="download-button">Télécharger la vidéo</button>
          </div>
        </div>
      </div>
    `;

    // Initialiser les références aux éléments
    this.progressBar = this.container.querySelector(".progress-fill");
    this.progressText = this.container.querySelector(".progress-text");
    this.queueValue = this.container.querySelector(".queue-value");
    this.queueContainer = this.container.querySelector(".queue-position-container");
    this.statusMessage = this.container.querySelector(".status-message");
    this.statusValue = this.container.querySelector(".status-value");
    this.timeValue = this.container.querySelector(".time-value");
    this.errorContainer = this.container.querySelector(".error-container");
    this.videoContainer = this.container.querySelector(".video-result-container");
    this.videoElement = this.container.querySelector(".result-video");
    this.downloadButton = this.container.querySelector(".download-button");

    // Configurer le bouton de téléchargement
    this.downloadButton.addEventListener('click', () => this.downloadVideo());

    // Démarrer le timer
    this.startElapsedTimeCounter();
  }

  startQueueSimulation() {
    // Décrémentation aléatoire de la file d'attente
    const updateQueue = () => {
      if (this.state.phase === 'queue' && this.state.queuePosition > 0) {
        // Réduire la position dans la file
        this.state.queuePosition = Math.max(0, this.state.queuePosition - 1);
        this.queueValue.textContent = this.state.queuePosition;

        // Mettre à jour la barre de progression (basée sur la position)
        const originalTotal = this.config.isPremium ?
          this.config.initialQueuePositionMax :
          this.config.initialQueuePositionMax * this.config.freeUserMultiplier;

        const progressPercent = 100 * (1 - (this.state.queuePosition / originalTotal));
        this.progressBar.style.width = `${Math.min(progressPercent, 100)}%`;

        // Si nous atteignons 0, passer à la phase de traitement
        if (this.state.queuePosition === 0) {
          this.state.phase = 'processing';
          this.queueContainer.style.display = 'none';
          this.updateStatusMessage();
        } else {
          // Programmer la prochaine mise à jour
          const updateInterval = Math.floor(
            Math.random() *
            (this.config.queueUpdateIntervalMax - this.config.queueUpdateIntervalMin + 1)
          ) + this.config.queueUpdateIntervalMin;

          setTimeout(updateQueue, updateInterval);
        }
      }

      // Changer le message de statut périodiquement
      this.updateStatusMessage();
    };

    // Démarrer le processus
    updateQueue();
  }

  updateStatusMessage() {
    let messages;

    // Sélectionner le groupe de messages selon la phase
    if (this.state.phase === 'queue') {
      messages = this.config.loadingMessages.queue;
    } else if (this.state.phase === 'processing') {
      messages = this.config.loadingMessages.processing;
    } else if (this.state.phase === 'finalizing') {
      messages = this.config.loadingMessages.finalizing;
    }

    // Mise à jour cyclique des messages
    if (messages && messages.length > 0) {
      this.state.messageIndex = (this.state.messageIndex + 1) % messages.length;
      this.statusMessage.textContent = messages[this.state.messageIndex];
    }

    // Mettre à jour la barre de progression pendant le traitement
    if (this.state.phase === 'processing' || this.state.phase === 'finalizing') {
      // Progression simulée basée sur le temps écoulé
      const elapsed = (Date.now() - this.state.startTime) / 1000; // en secondes
      let progressPercent;

      if (this.state.phase === 'processing') {
        // 20% à 80% pendant la phase de traitement
        progressPercent = 20 + Math.min(60, elapsed / 2);
      } else {
        // 80% à 95% pendant la finalisation
        progressPercent = 80 + Math.min(15, elapsed);
      }

      this.progressBar.style.width = `${progressPercent}%`;

      // Vérifier s'il faut passer à la phase de finalisation
      if (this.state.phase === 'processing' && progressPercent >= 80) {
        this.state.phase = 'finalizing';
      }
    }
  }

  startStatusChecking() {
    // Vérifier périodiquement le statut réel auprès de l'API
    const checkStatus = async () => {
      if (!this.config.taskId || this.state.phase === 'completed') return;

      try {
        const response = await fetch(`${this.config.statusEndpoint}${this.config.taskId}`);
        if (!response.ok) throw new Error("Erreur lors de la vérification du statut");

        const data = await response.json();

        if (data.success && data.data) {
          const status = data.data.status;
          this.state.realStatus = status;
          this.statusValue.textContent = this.translateStatus(status);

          // Si la vidéo est prête
          if (status === "Success" || status === "completed") {
            this.videoReady(data.data.video_url);
          } else if (status === "Fail" || status === "Failed") {
            this.handleError("La génération a échoué. Veuillez réessayer.");
          } else {
            // Continuer à vérifier
            setTimeout(checkStatus, this.config.checkInterval);
          }
        } else {
          throw new Error(data.error || "Erreur inconnue");
        }
      } catch (error) {
        console.error("Erreur lors de la vérification du statut:", error);
        // Continuer à vérifier malgré l'erreur
        setTimeout(checkStatus, this.config.checkInterval);
      }
    };

    // Démarrer le processus
    checkStatus();
  }

  translateStatus(status) {
    const statusMap = {
      'Queueing': 'En file d\'attente',
      'Preparing': 'En préparation',
      'Processing': 'En traitement',
      'Success': 'Terminé',
      'Fail': 'Échec',
      'completed': 'Terminé'
    };

    return statusMap[status] || status;
  }

  videoReady(videoUrl) {
    this.state.phase = 'completed';
    this.state.videoUrl = videoUrl;

    // Mettre à jour l'interface
    this.progressBar.style.width = '100%';
    this.progressText.textContent = 'Génération terminée!';
    this.statusMessage.textContent = 'Votre vidéo est prête!';

    // Afficher la vidéo
    this.videoElement.src = videoUrl;
    this.videoContainer.style.display = 'block';

    // Masquer les éléments de statut
    this.container.querySelector('.status-details').style.display = 'none';
  }

  handleError(message) {
    this.state.errorMessage = message;
    this.errorContainer.textContent = message;
    this.errorContainer.style.display = 'block';
    this.progressBar.classList.add('error');
  }

  downloadVideo() {
    if (!this.state.videoUrl) return;

    // Créer un lien temporaire pour télécharger
    const link = document.createElement('a');
    link.href = this.state.videoUrl;
    link.download = `video_${this.config.taskId}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  startElapsedTimeCounter() {
    // Afficher le temps écoulé
    const updateElapsedTime = () => {
      if (this.state.phase === 'completed') return;

      const elapsed = Math.floor((Date.now() - this.state.startTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;

      this.timeValue.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;

      setTimeout(updateElapsedTime, 1000);
    };

    updateElapsedTime();
  }
}

// Initialiser le système de file d'attente (exemple d'utilisation)
document.addEventListener('DOMContentLoaded', function() {
  // Vérifier si nous sommes sur une page de génération vidéo
  const statusContainer = document.getElementById('video-generation-status');
  if (statusContainer && statusContainer.dataset.taskId) {
    const queueSystem = new VideoQueueSimulator({
      containerId: 'video-generation-status',
      taskId: statusContainer.dataset.taskId,
      isPremium: statusContainer.dataset.premium === 'true',
      statusEndpoint: '/check_video_status/'
    });
  }
});