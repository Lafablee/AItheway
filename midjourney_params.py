
# Module pour gérer les paramètres Midjourney

class MidjourneyParams:
    # Formats d'aspect ratio supportés
    ASPECT_RATIOS = {
        "1:1": "--ar 1:1",  # Format carré (défaut)
        "16:9": "--ar 16:9",  # Format paysage (widescreen)
        "9:16": "--ar 9:16",  # Format portrait (mobile)
        "4:3": "--ar 4:3",  # Format standard
        "3:4": "--ar 3:4",  # Format portrait standard
        "2:1": "--ar 2:1",  # Format panoramique
        "1:2": "--ar 1:2",  # Format vertical allongé
        "3:2": "--ar 3:2",  # Format appareil photo
        "2:3": "--ar 2:3",  # Format portrait appareil photo
    }

    # Styles de rendu
    STYLES = {
        "raw": "--style raw",  # Style brut, peu d'interprétation artistique
        "cute": "--style cute",  # Style mignon, simplifié
        "scenic": "--style scenic",  # Style paysage, améliore les scènes naturelles
        "expressive": "--style expressive",  # Style expressif, plus artistique
    }

    # Niveaux de qualité
    QUALITY = {
        "1": "--quality 1",  # Qualité standard
        "2": "--quality 2",  # Haute qualité (par défaut)
    }

    # Niveaux de chaos (aléatoire/créativité)
    CHAOS_LEVELS = {
        "0": "--chaos 0",  # Aucun chaos
        "20": "--chaos 20",  # Bas niveau de chaos
        "50": "--chaos 50",  # Niveau moyen
        "100": "--chaos 100",  # Chaos élevé
    }

    # Versions modèles
    VERSIONS = {
        "6.0": "--v 6.0",  # Version 6.0 (par défaut)
        "5.2": "--v 5.2",  # Version 5.2
        "5.1": "--v 5.1",  # Version 5.1
        "5.0": "--v 5.0",  # Version 5.0
        "niji": "--niji 5",  # Version Niji (pour anime/style manga)
    }

    # Options de seed
    @staticmethod
    def seed(value):
        """Crée un paramètre seed pour la génération déterministe"""
        if value and str(value).isdigit():
            return f"--seed {value}"
        return ""

    # Options de style de crayon, peinture, etc.
    MEDIUMS = {
        "digital": ["digital art", "digital painting"],
        "traditional": ["oil painting", "watercolor", "acrylic painting"],
        "drawing": ["pencil drawing", "charcoal sketch", "ink drawing"],
        "print": ["linocut", "woodcut", "screenprint", "risograph"],
        "photographic": ["photography", "35mm film", "polaroid", "cinematic"]
    }

    @staticmethod
    def build_params(aspect_ratio=None, style=None, quality=None, chaos=None,
                     version=None, seed_value=None, no_text=False):
        """
        Construit une chaîne de paramètres Midjourney à partir des options sélectionnées

        Args:
            aspect_ratio (str): Clé du ratio d'aspect (ex: "16:9")
            style (str): Clé du style (ex: "raw")
            quality (str): Clé de qualité (ex: "2")
            chaos (str): Clé du niveau de chaos (ex: "50")
            version (str): Clé de la version (ex: "6.0")
            seed_value (int): Valeur de seed pour génération déterministe
            no_text (bool): Si True, ajoute le paramètre --no pour éviter le texte dans l'image

        Returns:
            str: Chaîne de paramètres formatée pour Midjourney
        """
        params = []

        if aspect_ratio and aspect_ratio in MidjourneyParams.ASPECT_RATIOS:
            params.append(MidjourneyParams.ASPECT_RATIOS[aspect_ratio])

        if style and style in MidjourneyParams.STYLES:
            params.append(MidjourneyParams.STYLES[style])

        if quality and quality in MidjourneyParams.QUALITY:
            params.append(MidjourneyParams.QUALITY[quality])

        if chaos and chaos in MidjourneyParams.CHAOS_LEVELS:
            params.append(MidjourneyParams.CHAOS_LEVELS[chaos])

        if version and version in MidjourneyParams.VERSIONS:
            params.append(MidjourneyParams.VERSIONS[version])

        if seed_value:
            seed_param = MidjourneyParams.seed(seed_value)
            if seed_param:
                params.append(seed_param)

        if no_text:
            params.append("--no text")

        return " ".join(params)

    @staticmethod
    def enhance_prompt(prompt, medium=None, enhance_details=False, photorealistic=False):
        """
        Améliore un prompt avec des détails supplémentaires et des styles

        Args:
            prompt (str): Le prompt de base
            medium (str): Clé du médium artistique (ex: "digital")
            enhance_details (bool): Si True, ajoute des paramètres de qualité
            photorealistic (bool): Si True, ajoute des paramètres de réalisme

        Returns:
            str: Prompt amélioré
        """
        additions = []

        # Ajouter un médium artistique si demandé
        if medium and medium in MidjourneyParams.MEDIUMS:
            options = MidjourneyParams.MEDIUMS[medium]
            # Choisir aléatoirement un style dans la catégorie
            import random
            selected_medium = random.choice(options)
            additions.append(selected_medium)

        # Ajouter des paramètres de qualité d'image si demandé
        if enhance_details:
            detail_phrases = ["highly detailed", "intricate details", "8K resolution",
                              "sharp focus", "sophisticated lighting"]
            # Ajouter 2 paramètres aléatoires de cette liste
            import random
            selected_details = random.sample(detail_phrases, 2)
            additions.extend(selected_details)

        # Ajouter des paramètres de réalisme si demandé
        if photorealistic:
            realism_phrases = ["photorealistic", "hyperrealistic", "ultra realistic",
                               "professional photography", "cinematic lighting"]
            # Ajouter 1 paramètre aléatoire de cette liste
            import random
            selected_realism = random.choice(realism_phrases)
            additions.append(selected_realism)

        # Combiner le prompt original avec les ajouts
        if additions:
            enhanced_prompt = f"{prompt}, {', '.join(additions)}"
            return enhanced_prompt

        return prompt