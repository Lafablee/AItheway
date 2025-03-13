/**
 * Right Sidebar Interactions
 * This script handles interactions for the image generation sidebar with tag selectors
 */
document.addEventListener('DOMContentLoaded', function() {
    // Handle aspect ratio changes and preview
    const ratioInputs = document.querySelectorAll('input[name="aspect_ratio"]');
    const ratioPreviewBox = document.getElementById('ratio-preview-box');
    const ratioPreviewInner = document.getElementById('ratio-preview-inner');

    // Update aspect ratio preview on load
    updateRatioPreview();

    // Listen for aspect ratio changes
    ratioInputs.forEach(input => {
        input.addEventListener('change', updateRatioPreview);
    });

    function updateRatioPreview() {
        const selectedRatio = document.querySelector('input[name="aspect_ratio"]:checked').value;
        const [width, height] = selectedRatio.split(':').map(Number);

        const boxWidth = ratioPreviewBox.clientWidth * 0.8;
        const boxHeight = ratioPreviewBox.clientHeight * 0.8;

        let previewWidth, previewHeight;

        if (width / height > boxWidth / boxHeight) {
            // Constrained by width
            previewWidth = boxWidth;
            previewHeight = boxWidth * (height / width);
        } else {
            // Constrained by height
            previewHeight = boxHeight;
            previewWidth = boxHeight * (width / height);
        }

        ratioPreviewInner.style.width = `${previewWidth}px`;
        ratioPreviewInner.style.height = `${previewHeight}px`;

        // Add a pulse animation to show the change
        ratioPreviewInner.classList.add('pulse-animation');
        setTimeout(() => {
            ratioPreviewInner.classList.remove('pulse-animation');
        }, 500);
    }

    // Handle creativity slider
    const creativitySlider = document.getElementById('creativity-slider');
    const creativityValue = document.getElementById('creativity-value');

    creativitySlider.addEventListener('input', function() {
        creativityValue.textContent = this.value;

        // Visual feedback as slider changes
        if (this.value > 70) {
            creativityValue.classList.add('high-value');
        } else {
            creativityValue.classList.remove('high-value');
        }
    });

    // Handle model change
    const modelInputs = document.querySelectorAll('input[name="model"]');

    modelInputs.forEach(input => {
        input.addEventListener('change', function() {
            // Update current model
            currentModel = this.value;

            // Show/hide model-specific options
            toggleModelSpecificOptions(currentModel);

            // Update token cost display
            updateTokenCost();

            // Pulse animation for token cost
            const tokenCost = document.querySelector('.token-cost');
            if (tokenCost) {
                tokenCost.classList.add('pulse-animation');
                setTimeout(() => {
                    tokenCost.classList.remove('pulse-animation');
                }, 500);
            }
        });
    });

    function toggleModelSpecificOptions(model) {
        const midjourneyOptions = document.querySelectorAll('.midjourney-only');
        const dalleOptions = document.querySelectorAll('.dalle-only');

        if (model === 'midjourney') {
            midjourneyOptions.forEach(option => option.style.display = 'block');
            dalleOptions.forEach(option => option.style.display = 'none');
        } else {
            midjourneyOptions.forEach(option => option.style.display = 'none');
            dalleOptions.forEach(option => option.style.display = 'block');
        }
    }

    // Initialize model specific options on load
    toggleModelSpecificOptions(document.querySelector('input[name="model"]:checked').value);

    // Add visual feedback for tag selection
    const tagInputs = document.querySelectorAll('.tag-option input');

    tagInputs.forEach(input => {
        input.addEventListener('change', function() {
            // Find all inputs with the same name
            const relatedInputs = document.querySelectorAll(`input[name="${this.name}"]`);

            // Remove active-tag class from all related labels
            relatedInputs.forEach(relatedInput => {
                const label = relatedInput.nextElementSibling;
                label.classList.remove('active-tag');
            });

            // Add active-tag class to selected label
            this.nextElementSibling.classList.add('active-tag');

            // Add a subtle animation to the sidebar
            document.querySelector('.generation__sidebar').classList.add('option-changed');
            setTimeout(() => {
                document.querySelector('.generation__sidebar').classList.remove('option-changed');
            }, 300);

            // Special handling for style changes
            if (this.name === 'style') {
                updatePromptSuggestion();
            }
        });
    });

    // Optional: Add prompt suggestions based on selected style
    function updatePromptSuggestion() {
        const selectedStyle = document.querySelector('input[name="style"]:checked');
        const promptInput = document.getElementById('prompt');

        if (!selectedStyle || !promptInput) return;

        let suggestion = '';
        const currentPrompt = promptInput.value.trim();

        // Only offer suggestions if the prompt is empty
        if (currentPrompt === '') {
            switch (selectedStyle.value) {
                case 'realistic':
                    suggestion = 'Une photo réaliste de...';
                    break;
                case 'cartoon':
                    suggestion = 'Un dessin animé coloré de...';
                    break;
                case 'anime':
                    suggestion = 'Un personnage d\'anime avec...';
                    break;
                case 'abstract':
                    suggestion = 'Une composition abstraite avec...';
                    break;
                case 'digital':
                    suggestion = 'Une illustration numérique de...';
                    break;
                case 'painting':
                    suggestion = 'Une peinture artistique de...';
                    break;
            }

            // Show suggestion tooltip
            const suggestionTooltip = document.createElement('div');
            suggestionTooltip.className = 'prompt-suggestion';
            suggestionTooltip.textContent = 'Suggestion: ' + suggestion;
            suggestionTooltip.style.position = 'absolute';
            suggestionTooltip.style.bottom = '100%';
            suggestionTooltip.style.left = '20px';
            suggestionTooltip.style.backgroundColor = 'var(--neltar-main-color)';
            suggestionTooltip.style.color = 'white';
            suggestionTooltip.style.padding = '8px 12px';
            suggestionTooltip.style.borderRadius = '5px';
            suggestionTooltip.style.fontSize = '12px';
            suggestionTooltip.style.marginBottom = '5px';
            suggestionTooltip.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
            suggestionTooltip.style.zIndex = '10';
            suggestionTooltip.style.pointerEvents = 'none';
            suggestionTooltip.style.opacity = '0';
            suggestionTooltip.style.transform = 'translateY(10px)';
            suggestionTooltip.style.transition = 'opacity 0.3s, transform 0.3s';

            const promptContainer = promptInput.parentNode;
            promptContainer.style.position = 'relative';
            promptContainer.appendChild(suggestionTooltip);

            // Show with animation
            setTimeout(() => {
                suggestionTooltip.style.opacity = '1';
                suggestionTooltip.style.transform = 'translateY(0)';
            }, 10);

            // Add a "Use Suggestion" button
            const useSuggestionBtn = document.createElement('button');
            useSuggestionBtn.textContent = 'Utiliser';
            useSuggestionBtn.style.marginLeft = '10px';
            useSuggestionBtn.style.background = 'white';
            useSuggestionBtn.style.color = 'var(--neltar-main-color)';
            useSuggestionBtn.style.border = 'none';
            useSuggestionBtn.style.borderRadius = '3px';
            useSuggestionBtn.style.padding = '2px 6px';
            useSuggestionBtn.style.fontSize = '11px';
            useSuggestionBtn.style.cursor = 'pointer';
            useSuggestionBtn.style.pointerEvents = 'all';

            suggestionTooltip.appendChild(useSuggestionBtn);

            useSuggestionBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                promptInput.value = suggestion;
                promptInput.focus();
                suggestionTooltip.remove();
            });

            // Remove after 6 seconds or when prompt gets focus
            setTimeout(() => {
                if (document.body.contains(suggestionTooltip)) {
                    suggestionTooltip.style.opacity = '0';
                    suggestionTooltip.style.transform = 'translateY(10px)';
                    setTimeout(() => suggestionTooltip.remove(), 300);
                }
            }, 6000);

            promptInput.addEventListener('focus', () => {
                if (document.body.contains(suggestionTooltip)) {
                    suggestionTooltip.style.opacity = '0';
                    suggestionTooltip.style.transform = 'translateY(10px)';
                    setTimeout(() => suggestionTooltip.remove(), 300);
                }
            }, { once: true });
        }
    }

    // Collect parameters when form is submitted
    const generateForm = document.getElementById('generate-image-form');
    if (generateForm) {
        generateForm.addEventListener('submit', function(event) {
            // Get all the parameter values
            const params = collectAllParameters();

            // Log parameters to console for debugging
            console.log('Submitting with parameters:', params);

            // Store parameters in form data or add as hidden inputs
            for (const key in params) {
                if (!this.querySelector(`input[name="${key}"]`)) {
                    const hiddenInput = document.createElement('input');
                    hiddenInput.type = 'hidden';
                    hiddenInput.name = key;
                    hiddenInput.value = params[key];
                    this.appendChild(hiddenInput);
                }
            }
        });
    }

    // Function to collect all parameters from UI
    function collectAllParameters() {
        const params = {};

        // Model
        params.model = document.querySelector('input[name="model"]:checked')?.value || 'dall-e';

        // Aspect ratio
        params.aspect_ratio = document.querySelector('input[name="aspect_ratio"]:checked')?.value || '1:1';

        // Style
        params.style = document.querySelector('input[name="style"]:checked')?.value || '';

        // Medium
        params.medium = document.querySelector('input[name="medium"]:checked')?.value || '';

        // Creativity level
        params.creativity = document.getElementById('creativity-slider')?.value || '50';

        // Quality
        params.quality = document.querySelector('input[name="quality"]:checked')?.value || '2';

        // Seed
        const seedValue = document.getElementById('seed-input')?.value;
        if (seedValue && seedValue.trim() !== '') {
            params.seed = seedValue;
        }

        // Toggles
        params.enhance_details = document.getElementById('enhance-details')?.checked || false;
        params.photorealistic = document.getElementById('photorealistic')?.checked || false;
        params.no_text = document.getElementById('no-text')?.checked || false;

        // Version
        params.version = document.querySelector('input[name="version"]:checked')?.value || '6.0';

        return params;
    }

    // Add CSS animations
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        @keyframes pulse-animation {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .active-tag {
            transform: scale(1.05);
        }
        
        .high-value {
            color: #f44336;
        }
        
        .option-changed {
            animation: sidebar-highlight 0.3s ease;
        }
        
        @keyframes sidebar-highlight {
            0% { background-color: var(--neltar-header-bg-color); }
            50% { background-color: rgba(124, 95, 227, 0.1); }
            100% { background-color: var(--neltar-header-bg-color); }
        }
    `;
    document.head.appendChild(styleElement);
});