// JavaScript pour la fonctionnalité du panneau de navigation gauche

document.addEventListener('DOMContentLoaded', function() {
    // Gestion des menus déroulants
    const dropdownToggles = document.querySelectorAll('.dropdown_toggle');

    dropdownToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const parent = this.closest('.dropdown_menu');

            // Fermer tous les autres menus déroulants
            document.querySelectorAll('.dropdown_menu.open').forEach(menu => {
                if (menu !== parent) {
                    menu.classList.remove('open');
                }
            });

            // Basculer l'état du menu actuel
            parent.classList.toggle('open');
        });
    });

    // Gestion du panneau latéral en mode responsive
    const mobileMenuToggle = document.querySelector('.mobile_menu_toggle');
    const leftPanel = document.querySelector('.neltar_fn_leftpanel');

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function() {
            leftPanel.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
    }

    // Fermer le menu lors du clic en dehors sur les petits écrans
    document.addEventListener('click', function(event) {
        const isSmallScreen = window.innerWidth <= 992;
        const isClickInsideMenu = leftPanel.contains(event.target);
        const isClickOnToggle = mobileMenuToggle && mobileMenuToggle.contains(event.target);

        if (isSmallScreen && !isClickInsideMenu && !isClickOnToggle && leftPanel.classList.contains('active')) {
            leftPanel.classList.remove('active');
            document.body.classList.remove('menu-open');
        }
    });

    // Ajuster le contenu principal en fonction de la largeur du panneau latéral
    function adjustMainContent() {
        const mainContent = document.querySelector('.neltar_fn_content');
        if (!mainContent) return;

        if (window.innerWidth > 992) {
            mainContent.style.marginLeft = '280px';
        } else {
            mainContent.style.marginLeft = '60px';
        }

        // En mode mobile, pas de marge
        if (window.innerWidth <= 576) {
            mainContent.style.marginLeft = '0';
        }
    }

    // Appliquer au chargement et au redimensionnement
    adjustMainContent();
    window.addEventListener('resize', adjustMainContent);
});