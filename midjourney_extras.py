# midjourney_extras.py
# Module pour les fonctionnalités supplémentaires de Midjourney

import re
import json
import numpy as np
from PIL import Image
import io
import requests
from flask import current_app as app
import os
import time
from datetime import datetime


class MidjourneyAnalyzer:
    """Classe pour analyser et extraire des informations des images générées par Midjourney"""

    @staticmethod
    def extract_seed_from_prompt(prompt):
        """Extrait la valeur de seed d'un prompt Midjourney"""
        seed_pattern = r'--seed\s+(\d+)'
        match = re.search(seed_pattern, prompt)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def extract_ratio_from_prompt(prompt):
        """Extrait le ratio d'aspect d'un prompt Midjourney"""
        ratio_pattern = r'--ar\s+(\d+):(\d+)'
        match = re.search(ratio_pattern, prompt)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return f"{width}:{height}"
        return None

    @staticmethod
    def extract_parameters(prompt):
        """Extrait tous les paramètres d'un prompt Midjourney"""
        params = {}

        # Extraire le ratio d'aspect
        ratio = MidjourneyAnalyzer.extract_ratio_from_prompt(prompt)
        if ratio:
            params['aspect_ratio'] = ratio

        # Extraire la seed
        seed = MidjourneyAnalyzer.extract_seed_from_prompt(prompt)
        if seed:
            params['seed'] = seed

        # Extraire le style
        style_pattern = r'--style\s+(\w+)'
        style_match = re.search(style_pattern, prompt)
        if style_match:
            params['style'] = style_match.group(1)

        # Extraire la qualité
        quality_pattern = r'--quality\s+(\d+)'
        quality_match = re.search(quality_pattern, prompt)
        if quality_match:
            params['quality'] = quality_match.group(1)

        # Extraire le chaos
        chaos_pattern = r'--chaos\s+(\d+)'
        chaos_match = re.search(chaos_pattern, prompt)
        if chaos_match:
            params['chaos'] = chaos_match.group(1)

        # Extraire la version
        version_pattern = r'--v\s+(\d+\.\d+)'
        version_match = re.search(version_pattern, prompt)
        if version_match:
            params['version'] = version_match.group(1)

        # Vérifier si --no est utilisé
        no_text_pattern = r'--no\s+text'
        if re.search(no_text_pattern, prompt):
            params['no_text'] = True

        return params

    @staticmethod
    def clean_prompt(prompt):
        """Nettoie un prompt Midjourney en enlevant tous les paramètres"""
        # Liste des paramètres à enlever
        param_patterns = [
            r'--ar\s+\d+:\d+',
            r'--seed\s+\d+',
            r'--style\s+\w+',
            r'--quality\s+\d+',
            r'--chaos\s+\d+',
            r'--v\s+\d+\.\d+',
            r'--no\s+text',
            r'--niji\s+\d+'
        ]

        # Enlever chaque paramètre
        clean_prompt = prompt
        for pattern in param_patterns:
            clean_prompt = re.sub(pattern, '', clean_prompt)

        # Nettoyer les espaces multiples
        clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()

        return clean_prompt


class MidjourneyImageUtils:
    """Outils utilitaires pour les images générées par Midjourney"""

    @staticmethod
    def download_image(url, token=None):
        """Télécharge une image à partir d'une URL"""
        try:
            headers = {}
            if token:
                # Vérifier si l'URL contient déjà des paramètres
                separator = '&' if '?' in url else '?'
                url = f"{url}{separator}token={token}"

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            app.logger.error(f"Error downloading image: {str(e)}")
            return None

    @staticmethod
    def get_image_dimensions(image_url, token=None):
        """Récupère les dimensions d'une image"""
        try:
            img = MidjourneyImageUtils.download_image(image_url, token)
            if img:
                return img.size
            return None
        except Exception as e:
            app.logger.error(f"Error getting image dimensions: {str(e)}")
            return None

    @staticmethod
    def create_variation_image(image_url, seed_modifier=1000, token=None, prompt=None, aspect_ratio=None):
        """
        Crée une variation d'une image existante en modifiant la seed
        Note: Cette fonction est un placeholder pour démontrer une possible feature
        Dans une implémentation réelle, cela serait intégré avec l'API Midjourney
        """
        # Extraire les paramètres de l'image originale
        params = {}
        if prompt:
            params = MidjourneyAnalyzer.extract_parameters(prompt)

        # Modifier la seed
        original_seed = params.get('seed', int(time.time() % 10000))
        new_seed = original_seed + seed_modifier

        # Utiliser le même ratio que l'original ou celui spécifié
        ratio = aspect_ratio or params.get('aspect_ratio', '1:1')

        # Construire un nouveau prompt
        clean_prompt = MidjourneyAnalyzer.clean_prompt(prompt) if prompt else "Image variation"
        new_params = f"--seed {new_seed} --ar {ratio}"

        return {
            'prompt': f"{clean_prompt} {new_params}",
            'seed': new_seed,
            'aspect_ratio': ratio,
            # Dans une implémentation réelle, vous pourriez ici déclencher une génération
            # ou retourner des données pour que l'interface utilisateur puisse déclencher la génération
        }


class MidjourneyTemplateSystem:
    """Système de modèles pour générer des prompts efficaces pour Midjourney"""

    # Modèles de prompts pour différents types d'images
    TEMPLATES = {
        'portrait': {
            'template': "{subject}, portrait, {style}, {lighting}, {background}, {camera}",
            'options': {
                'style': ['realistic', 'cinematic', 'studio quality', 'detailed', 'photorealistic'],
                'lighting': ['soft lighting', 'dramatic lighting', 'rim lighting', 'natural lighting',
                             'studio lighting'],
                'background': ['simple background', 'blurred background', 'studio background', 'nature background',
                               'neutral background'],
                'camera': ['85mm lens', 'portrait lens', 'professional camera', 'DSLR', 'high resolution']
            }
        },
        'landscape': {
            'template': "{location}, {time_of_day}, {weather}, {style}, wide angle",
            'options': {
                'time_of_day': ['sunrise', 'sunset', 'golden hour', 'blue hour', 'midday', 'twilight', 'night'],
                'weather': ['clear sky', 'cloudy', 'foggy', 'misty', 'stormy', 'rainy', 'snowy'],
                'style': ['panoramic', 'cinematic', 'epic', 'wide shot', 'dramatic', 'photorealistic']
            }
        },
        'product': {
            'template': "{product}, product photography, studio lighting, {background}, {angle}, commercial, advertising",
            'options': {
                'background': ['white background', 'gradient background', 'studio background', 'minimalist background',
                               'contextual background'],
                'angle': ['front view', '3/4 view', 'top-down view', 'close-up', 'detail shot']
            }
        },
        'concept_art': {
            'template': "{subject}, concept art, {style}, {artist}, detailed, professional",
            'options': {
                'style': ['fantasy', 'sci-fi', 'cyberpunk', 'steampunk', 'futuristic', 'medieval', 'post-apocalyptic'],
                'artist': ['by artstation', 'by top artists', 'trending on artstation', 'professional', 'detailed']
            }
        },
        'food': {
            'template': "{dish}, food photography, {lighting}, {angle}, {style}, appetizing, delicious",
            'options': {
                'lighting': ['soft lighting', 'natural lighting', 'warm lighting', 'studio lighting', 'side lighting'],
                'angle': ['top-down', '45-degree angle', 'close-up', 'side view', 'three-quarter view'],
                'style': ['commercial', 'editorial', 'rustic', 'minimalist', 'gourmet']
            }
        }
    }

    @staticmethod
    def generate_from_template(template_key, subject=None, **kwargs):
        """
        Génère un prompt à partir d'un modèle

        Args:
            template_key (str): Clé du modèle à utiliser (portrait, landscape, etc.)
            subject (str): Sujet principal de l'image
            **kwargs: Paramètres spécifiques pour le modèle

        Returns:
            str: Prompt formaté pour Midjourney
        """
        if template_key not in MidjourneyTemplateSystem.TEMPLATES:
            return subject or "Please provide a detailed prompt"

        template_data = MidjourneyTemplateSystem.TEMPLATES[template_key]
        template = template_data['template']
        options = template_data['options']

        # Initialiser les valeurs par défaut
        values = {'subject': subject or f"{template_key} subject"}

        # Pour chaque option, utiliser la valeur fournie ou en choisir une aléatoire
        import random
        for key, choices in options.items():
            if key in kwargs and kwargs[key]:
                values[key] = kwargs[key]
            else:
                values[key] = random.choice(choices)

        # Formatter le template avec les valeurs
        return template.format(**values)


class MidjourneyCollection:
    """Gestion des collections d'images Midjourney"""

    @staticmethod
    def create_collection(redis_client, user_id, name, description=""):
        """Crée une nouvelle collection pour un utilisateur"""
        collection_id = f"collection:{user_id}:{int(time.time())}"

        # Créer la collection
        collection_data = {
            'id': collection_id,
            'user_id': user_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'images': []
        }

        # Stocker dans Redis
        redis_client.setex(
            collection_id,
            60 * 60 * 24 * 30,  # 30 jours
            json.dumps(collection_data)
        )

        # Ajouter à l'index des collections de l'utilisateur
        user_collections_key = f"user:{user_id}:collections"
        redis_client.sadd(user_collections_key, collection_id)

        return collection_id

    @staticmethod
    def add_image_to_collection(redis_client, collection_id, image_key, user_id=None):
        """Ajoute une image à une collection"""
        # Récupérer la collection
        collection_data = redis_client.get(collection_id)
        if not collection_data:
            return False

        collection = json.loads(collection_data)

        # Vérifier que la collection appartient à l'utilisateur si un user_id est fourni
        if user_id and collection.get('user_id') != str(user_id):
            return False

        # Ajouter l'image si elle n'est pas déjà présente
        if image_key not in collection['images']:
            collection['images'].append(image_key)
            collection['updated_at'] = datetime.now().isoformat()

            # Mettre à jour la collection
            redis_client.setex(
                collection_id,
                60 * 60 * 24 * 30,  # 30 jours
                json.dumps(collection)
            )

            return True

        return False

    @staticmethod
    def get_user_collections(redis_client, user_id):
        """Récupère toutes les collections d'un utilisateur"""
        user_collections_key = f"user:{user_id}:collections"
        collection_ids = redis_client.smembers(user_collections_key)

        collections = []
        for collection_id in collection_ids:
            collection_data = redis_client.get(collection_id)
            if collection_data:
                collections.append(json.loads(collection_data))

        # Trier par date de création (plus récent en premier)
        collections.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return collections

    @staticmethod
    def get_collection(redis_client, collection_id, user_id=None):
        """Récupère une collection spécifique"""
        collection_data = redis_client.get(collection_id)
        if not collection_data:
            return None

        collection = json.loads(collection_data)

        # Vérifier que la collection appartient à l'utilisateur si un user_id est fourni
        if user_id and collection.get('user_id') != str(user_id):
            return None

        return collection

