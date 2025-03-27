import os
import traceback
from http.client import responses

import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, send_file, abort, redirect
from io import BytesIO
from PIL import Image
from openai import images
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import requests
import jwt
from jwt import InvalidTokenError
from functools import wraps
import psycopg2
from datetime import datetime, timedelta
import time
from urllib.parse import quote
import redis
import json
import subprocess

from ai_models import create_ai_manager
from audio_manager import AudioManager
from video_manager import VideoManager

from tasks import start_background_tasks
import uuid
import asyncio
from pathlib import Path
from storage_manager import FileStorage, StorageManager
from storage_tasks import migrate_old_redis_content

from config import redis_client, TEMP_STORAGE_DURATION, PERMANENT_STORAGE_DURATION



# Définir les chemins des dossiers de templates
template_dir = os.path.abspath('templates')
template_temp_dir = os.path.abspath('templates_temp')


# Configuration du stockage fichier
STORAGE_BASE_PATH = os.getenv("STORAGE_BASE_PATH", os.path.join(os.getcwd(), 'storage'))
file_storage = FileStorage(STORAGE_BASE_PATH)

test_mode = os.getenv("STORAGE_TEST_MODE", "false").lower() == "true"
test_ttl_minutes = int(os.getenv("STORAGE_TEST_TTL_MINUTES", "10"))

if test_mode:
    # 10 minutes en mode test au lieu de 24 heures
    temp_duration = timedelta(minutes=test_ttl_minutes)
    print(f"Mode TEST activé: TTL initial réduit à {test_ttl_minutes} minutes")
else:
    # Valeur normale pour le mode production
    temp_duration = timedelta(seconds=TEMP_STORAGE_DURATION) if isinstance(TEMP_STORAGE_DURATION,int) else TEMP_STORAGE_DURATION

# Créer le gestionnaire de stockage unifié
storage_manager = StorageManager(
    redis_client=redis_client,
    file_storage=file_storage,
    temp_duration=temp_duration
)


load_dotenv(dotenv_path='.env')

openai.api_key = os.getenv("openai.api_key")
SECRET_KEY = os.getenv('SECRET_KEY')
FIXED_TOKEN = os.getenv('FIXED_TOKEN')
LOGIN_URL = "https://aitheway.com/login/"

app = Flask(__name__)

app.jinja_loader.searchpath = [template_dir, template_temp_dir]  # Ajouter les deux dossiers

# Création du gestionnaire de modèles AI
ai_manager = create_ai_manager()

# Configuration des fichiers uploadés
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit
FFMPEG_TIMEOUT = 30  # secondes
MAX_VIDEO_SIZE_FOR_THUMBNAIL = 500 * 1024 * 1024  # 500 MB

deep_ai_api_key = os.getenv('DEEPAI_API_KEY')

# Configuration Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # Important pour les données binaires
)
from midjourney_extras import (
    MidjourneyAnalyzer,
    MidjourneyImageUtils,
    MidjourneyTemplateSystem,
    MidjourneyCollection
)
from gallery_manager import GalleryManager

# Durées de conservation
TEMP_STORAGE_DURATION = timedelta(hours=24)  # Pour les uploads temporaires
PERMANENT_STORAGE_DURATION = timedelta(days=30)  # Pour les images sauvegardées

# Constantes pour les types d'abonnements
SUBSCRIPTION_TYPES = {
    "BASIC_FREE": "basic_free",
    "BASIC_PAID": "basic_paid",
}

# Mappage des IDs d'abonnement vers les types
SUBSCRIPTION_ID_MAPPING = {
    29094: SUBSCRIPTION_TYPES["BASIC_FREE"],  # Nouveau plan gratuit
    28974: SUBSCRIPTION_TYPES["BASIC_PAID"],  # Ancien plan gratuit - maintenat payant
}

# Nombre de tokens par type d'abonnement
TOKEN_ALLOCATION = {
    SUBSCRIPTION_TYPES["BASIC_FREE"]: 150,
    SUBSCRIPTION_TYPES["BASIC_PAID"]: 500,
}

#période de rechargement  en jours
REFILL_PERIODS = {
    SUBSCRIPTION_TYPES["BASIC_FREE"]: 0.01, # ~15 minutes au lieu de 3 jours
    SUBSCRIPTION_TYPES["BASIC_PAID"]: 0.02 # ~30 minutes au lieu de 30 jours
}

GRACE_PERIOD = 0.04 # # ~1 heures au lieu de 60 jours

# Permissions par type d'abonnement
PERMISSIONS = {
    SUBSCRIPTION_TYPES["BASIC_FREE"]: ["view_content", "generate_image", "upload_enhance", "generate_audio", "generate_video"],
    SUBSCRIPTION_TYPES["BASIC_PAID"]: ["view_content", "generate_image", "upload_enhance", "generate_audio", "generate_video"],
}

class ImageManager:
    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.prefix = "img"

    def _generate_key(self, user_id, type="temp"):
        """Génère une clé unique avec préfixe"""

        timestamp = datetime.now().timestamp()
        return f"img:temp:image:{user_id}:{timestamp}"

    def store_temp_image(self, user_id, image_data, metadata=None):
        """Version améliorée du stockage"""
        try:
            # Génération d'une clé unique
            key = self._generate_key(user_id)
            app.logger.error(f"Generated key for image: {key}")

            # Convertir les métadonnées dict en format Redis si nécessaire
            if metadata and isinstance(metadata, dict):
                redis_metadata = {
                    k: v.encode('utf-8') if isinstance(v, str) else v
                    for k, v in metadata.items()
                }
            else:
                redis_metadata = metadata

            # Stocker via le gestionnaire de stockage
            self.storage.store(key, image_data, redis_metadata, 'images')

            # Ajouter à l'index utilisateur
            user_index_key = f"user:{user_id}:images"
            self.storage.redis.lpush(user_index_key, key)

            return key

        except Exception as e:
            app.logger.error(f"Redis storage error: {str(e)}")
            app.logger.error(f"Failed to store image with metadata: {metadata}")
            raise

    def get_user_history(self, user_id, history_type="enhanced"):
        """Récupère l'historique des images d'un utilisateur"""
        app.logger.error(f"Fetching history for user {user_id} with type {history_type}")

        # Les patterns restent les mêmes car on interroge toujours Redis pour les clés
        patterns = [
            f"img:temp:image:{user_id}:*",
            "img:temp:image:midjourney:*"
        ]
        images = []
        processed_groups = set()

        # Récupérer toutes les clés depuis Redis (pas besoin de changer cette partie)
        all_keys = []
        for pattern in patterns:
            keys = self.storage.redis.keys(pattern)
            all_keys.extend(keys)

        app.logger.error(f"Found {len(all_keys)} image keys in Redis")

        # Filtrer pour ne garder que les clés d'images (pas les métadonnées)
        image_keys = [k for k in all_keys if not (isinstance(k, bytes) and k.endswith(b':meta')) and
                                        not (isinstance(k, str) and k.endswith(':meta'))]
        app.logger.error(f"Filtered image keys: {image_keys}")

        # Traitement spécial pour les images générées par Midjourney
        if history_type == "generated":
            user_midjourney_key = f"user:{user_id}:midjourney_history"
            midjourney_task_ids = self.storage.redis.smembers(user_midjourney_key)

            if midjourney_task_ids:
                # Convertir les task_ids en bytes si nécessaire
                task_ids = [task_id.decode('utf-8') if isinstance(task_id, bytes) else task_id
                            for task_id in midjourney_task_ids]

                app.logger.error(f"Found Midjourney tasks for user {user_id}: {task_ids}")

                # Pour chaque tâche Midjourney associée à l'utilisateur
                for task_id in task_ids:
                    # Vérifier si on a déjà traité ce groupe
                    if task_id in processed_groups:
                        continue

                    # Récupérer les données du groupe Midjourney
                    group_key = f"midjourney_group:{task_id}"

                    # Ici, on utilise le StorageManager pour récupérer les données du groupe
                    # Le format est différent car StorageManager normalise les données
                    group_data = self.storage.get(group_key, 'metadata')

                    if group_data:
                        try:
                            # Si les données sont binaires, les décoder
                            if isinstance(group_data, bytes):
                                group = json.loads(group_data.decode('utf-8'))
                            else:
                                group = json.loads(group_data)

                            # Vérifier si le groupe est complet
                            if group and (group.get('status') == 'completed' or group.get('status') == 'complete'):
                                # S'assurer que le groupe a des images
                                if len(group.get('images', [])) > 0:
                                    processed_groups.add(task_id)

                                    # Formatage des URLs pour chaque image du groupe
                                    formatted_images = []
                                    for img in group['images']:
                                        # Générer la clé si elle n'existe pas
                                        img_key = img.get('key')
                                        if not img_key:
                                            variation = img.get('variation_number') or img.get('choice') or 0
                                            img_key = f"midjourney_image:{task_id}:{variation}"

                                        formatted_images.append({
                                            'url': f"/image/{img_key}",
                                            'key': img_key,
                                            'number': img.get('variation_number') or img.get('choice') or 0
                                        })

                                    image_data = {
                                        'model': 'midjourney',
                                        'task_id': task_id,
                                        'prompt': group.get('prompt', 'Unknown prompt'),
                                        'timestamp': group.get('timestamp', datetime.now().isoformat()),
                                        'images': formatted_images,
                                        'status': 'completed'
                                    }
                                    images.append(image_data)
                                    app.logger.error(f"Added Midjourney group from user index: {image_data}")
                        except json.JSONDecodeError as e:
                            app.logger.error(f"Error parsing Midjourney group data: {e}")
                            continue

        # Traitement des images individuelles
        for key in image_keys:
            try:
                # Convertir la clé en string si nécessaire
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key

                # Récupérer les métadonnées via le StorageManager
                metadata = self.storage.get_metadata(key_str)
                app.logger.error(f"Checking metadata for key {key_str}: {metadata}")

                # Vérifier si l'image existe toujours
                if not self.storage.get(key_str, 'images'):
                    app.logger.error(f"Image {key_str} no longer exists, skipping")
                    continue

                # Ici, les métadonnées sont déjà normalisées par StorageManager (pas de bytes)
                image_type = metadata.get('type') if metadata else None

                # Traitement selon le type d'historique demandé
                if history_type == "generated" and image_type == 'generated':
                    model = metadata.get('model', '')

                    if model == 'midjourney':
                        task_id = metadata.get('task_id', '')

                        # Éviter de traiter plusieurs fois le même groupe
                        if task_id in processed_groups:
                            continue

                        group = self.get_midjourney_group(task_id)
                        if group and group.get('status') == 'completed':
                            # S'assurer que le groupe a bien 4 images
                            if len(group.get('images', [])) == 4:
                                processed_groups.add(task_id)

                                # Formatage des URLs pour chaque image du groupe
                                formatted_images = []
                                for img in group['images']:
                                    formatted_images.append({
                                        'url': f"/image/{img['key']}",
                                        'key': img['key'],
                                        'number': img['number']
                                    })

                                image_data = {
                                    'model': 'midjourney',
                                    'task_id': task_id,
                                    'prompt': group['prompt'],
                                    'timestamp': group['timestamp'],
                                    'images': formatted_images,
                                    'status': 'completed'
                                }
                                images.append(image_data)
                                app.logger.error(f"Added Midjourney group: {image_data}")

                    else:
                        try:
                            # Récupérer les valeurs des métadonnées
                            decoded_prompt = metadata.get('prompt', '')
                            decoded_timestamp = metadata.get('timestamp', '')
                            model = metadata.get('model', 'unknown')

                            # Vérifier que l'image appartient à l'utilisateur pour les images DALL-E
                            # Pour Midjourney, on accepte toutes les images car elles sont partagées
                            if model != 'midjourney' and f"img:temp:image:{user_id}" not in key_str:
                                continue

                            # Récupérer les paramètres (déjà en string)
                            try:
                                parameters_str = metadata.get('parameters', '{}')
                                parameters = json.loads(parameters_str) if isinstance(parameters_str,
                                                                                      str) else parameters_str
                            except (json.JSONDecodeError, TypeError):
                                app.logger.error(f"Error parsing parameters for key {key}")
                                parameters = {}

                            # Construire l'objet image
                            image_data = {
                                'key': key_str,
                                'url': f"/image/{key_str}",
                                'prompt': decoded_prompt,
                                'timestamp': decoded_timestamp,
                                'parameters': parameters,
                                'model': model
                            }

                            # Ajouter des champs spécifiques à Midjourney si présents
                            if model == 'midjourney':
                                image_data['task_id'] = metadata.get('task_id', '')
                                image_data['make_task_id'] = metadata.get('make_task_id', '')

                            images.append(image_data)
                            app.logger.error(f"Added generated image to history: {image_data}")

                        except Exception as e:
                            app.logger.error(f"Error processing generated image metadata: {e}")
                            continue

                elif history_type == "enhanced" and image_type == 'enhanced':
                    try:
                        # Traitement spécifique pour les images améliorées
                        original_key = metadata.get('original_key')
                        timestamp = metadata.get('timestamp', '')

                        enhanced_data = {
                            'enhanced_key': key_str,
                            'original_key': original_key if original_key else None,
                            'timestamp': timestamp,
                            'enhanced_url': f"/image/{key_str}",
                            'original_url': f"/image/{original_key}" if original_key else None
                        }

                        images.append(enhanced_data)
                        app.logger.error(f"Added enhanced image to history: {enhanced_data}")

                    except Exception as e:
                        app.logger.error(f"Error processing enhanced image metadata: {e}")
                        continue

            except Exception as e:
                app.logger.error(f"Error processing image key {key}: {e}")
                continue

        # Récupération depuis le disque
        metadata_dir = os.path.join(self.storage.file_storage.base_path, 'metadata')
        disk_count = 0

        # Parcourir récursivement les fichiers de métadonnées
        for root, dirs, files in os.walk(metadata_dir):
            for file in files:
                try:
                    filepath = os.path.join(root, file)

                    with open(filepath, 'r') as f:
                        metadata = json.load(f)

                    # Vérifier si ces métadonnées correspondent à une image de cet utilisateur
                    original_key = metadata.get('original_redis_key', '')

                    # Déterminer si cette image correspond au type d'historique demandé
                    matches_type = False

                    if history_type == "enhanced" and metadata.get('type') == 'enhanced':
                        matches_type = True
                    elif history_type == "generated" and metadata.get('type') == 'generated':
                        matches_type = True

                    # Vérifier si l'image appartient à l'utilisateur
                    belongs_to_user = False
                    if metadata.get('user_id') == str(user_id) or metadata.get('user_id') == user_id:
                        belongs_to_user = True
                    elif original_key.startswith(f"img:temp:image:{user_id}:"):
                        belongs_to_user = True

                    if matches_type and belongs_to_user:
                        # Pour les images Midjourney, gérer les groupes
                        if metadata.get('model') == 'midjourney':
                            task_id = metadata.get('task_id')
                            if task_id and task_id not in processed_groups:
                                # Chercher toutes les images de ce groupe
                                midjourney_group = self._find_midjourney_group_on_disk(task_id, metadata_dir)
                                if midjourney_group:
                                    processed_groups.add(task_id)
                                    images.append(midjourney_group)
                        else:
                            # Pour les images standard
                            key_hash = os.path.basename(file)

                            # Vérifier que le fichier image existe bien
                            image_path = os.path.join(
                                self.storage.file_storage.base_path,
                                'images',
                                key_hash[:2],
                                key_hash[2:4],
                                key_hash
                            )

                            if os.path.exists(image_path):
                                image_data = self._create_image_history_entry(original_key, metadata)
                                images.append(image_data)
                                disk_count += 1

                except Exception as e:
                    app.logger.error(f"Error processing metadata file {file}: {str(e)}")

        app.logger.error(f"Found {disk_count} additional image files on disk for user {user_id}")

        # Trier les images par timestamp (plus récent en premier)
        sorted_images = sorted(images, key=lambda x: x['timestamp'], reverse=True)
        app.logger.error(f"Returning {len(sorted_images)} sorted images")

        return sorted_images

    def _find_midjourney_group_on_disk(self, task_id, metadata_dir):
        """Trouve toutes les images d'un groupe Midjourney stockées sur disque"""

        group_images = []
        prompt = ""
        timestamp = ""

        # Parcourir les métadonnées pour trouver toutes les images du groupe
        for root, dirs, files in os.walk(metadata_dir):
            for file in files:
                try:
                    filepath = os.path.join(root, file)

                    with open(filepath, 'r') as f:
                        metadata = json.load(f)

                    # Vérifier si c'est une image Midjourney de ce groupe
                    if (metadata.get('model') == 'midjourney' and
                            metadata.get('task_id') == task_id):

                        # Récupérer les infos communes au groupe
                        if not prompt and metadata.get('prompt'):
                            prompt = metadata.get('prompt')
                        if not timestamp and metadata.get('timestamp'):
                            timestamp = metadata.get('timestamp')

                        # Récupérer la clé et vérifier que le fichier existe
                        key_hash = os.path.basename(file)
                        original_key = metadata.get('original_redis_key', '')
                        variation_number = metadata.get('variation_number', 0)

                        image_path = os.path.join(
                            self.storage.file_storage.base_path,
                            'images',
                            key_hash[:2],
                            key_hash[2:4],
                            key_hash
                        )

                        if os.path.exists(image_path):
                            group_images.append({
                                'url': f"/image/{original_key}",
                                'key': original_key,
                                'number': variation_number
                            })

                except Exception as e:
                    app.logger.error(f"Error processing Midjourney metadata file {file}: {str(e)}")

        # Si on a trouvé des images, créer l'entrée de groupe
        if group_images:
            return {
                'model': 'midjourney',
                'task_id': task_id,
                'prompt': prompt,
                'timestamp': timestamp,
                'images': group_images,
                'status': 'completed'
            }

        return None

    def _create_image_history_entry(self, key, metadata):
        """Crée une entrée d'historique pour une image standard"""
        if metadata.get('type') == 'generated':
            # Pour les images générées
            return {
                'key': key,
                'url': f"/image/{key}",
                'prompt': metadata.get('prompt', ''),
                'timestamp': metadata.get('timestamp', ''),
                'parameters': json.loads(metadata.get('parameters', '{}')) if isinstance(metadata.get('parameters'),
                                                                                         str) else metadata.get(
                    'parameters', {}),
                'model': metadata.get('model', 'unknown')
            }
        elif metadata.get('type') == 'enhanced':
            # Pour les images améliorées
            return {
                'enhanced_key': key,
                'original_key': metadata.get('original_key'),
                'timestamp': metadata.get('timestamp', ''),
                'enhanced_url': f"/image/{key}",
                'original_url': f"/image/{metadata.get('original_key')}" if metadata.get('original_key') else None
            }
        else:
            # Format par défaut
            return {
                'key': key,
                'url': f"/image/{key}",
                'timestamp': metadata.get('timestamp', ''),
                'type': metadata.get('type', 'unknown')
            }

    def save_image(self, user_id, image_data):
        """Sauvegarde une image de manière permanente"""
        key = f"user:{user_id}:images:{datetime.now().timestamp()}"

        # Stocker directement sur le disque en contournant Redis
        self.storage.file_storage.store_file(key, image_data, 'images')
        return key

    def get_image(self, key):
        """Récupère une image depuis n'importe quel stockage"""
        return self.storage.get(key, 'images')

    def delete_image(self, key):
        """Suppression propre avec métadonnées"""
        return self.storage.delete(key, 'images')

    def get_image_metadata(self, key):
        """Récupère les métadonnées d'une image"""
        return self.storage.get_metadata(key)

    def create_midjourney_group(self, task_id, prompt):
        group_key = f"midjourney_group:{task_id}"
        group_data = self.redis.get(group_key)
        if not group_data:
            return None
        return json.loads(group_data[b'data'].decode('utf-8'))

    def add_to_midjourney_group(self, task_id, image_data):
        group_key = f"{self.prefix}:midjourney:group:{task_id}"
        group_data = self.redis.get(group_key)

        if not group_data:
            app.logger.error(f"Group key {group_key} not found for task_id:{task_id}")
            return None

        group = json.loads(group_data)
        group['images'].append(image_data)

        # Mise à jour du statut
        if len(group['images']) == 4:
            group['status'] = 'complete'
        else:
            group['status'] = 'partial'

        self.redis.setex(
            group_key,
            TEMP_STORAGE_DURATION,
            json.dumps(group)
        )
        app.logger.error(f"Added image to group {task_id}. Status: {group['status']}")
        return group

    def get_midjourney_task_status(self, task_id):
        """Récupère le statut d'une tâche Midjourney"""
        metadata_key = f"midjourney_task:{task_id}"
        group_key = f"midjourney_group:{task_id}"

        # Récupérer les métadonnées de la tâche
        task_data = self.redis.hgetall(metadata_key)
        group_data = self.redis.get(group_key)

        if not task_data:
            return None

        status_info = {
            'task_id': task_id,
            'status': task_data.get(b'status', b'unknown').decode('utf-8'),
            'prompt': task_data.get(b'prompt', b'').decode('utf-8'),
        }

        if group_data:
            group_info = json.loads(group_data)
            status_info.update({
                'initial_grid': group_info.get('initial_grid'),
                'images': group_info.get('images', []),
                'group_status': group_info.get('status')
            })

        return status_info

    def store_midjourney_upscale(self, task_id, image_url, upscale_number):
        """Stocke une image upscale de Midjourney"""
        group_key = f"midjourney_group:{task_id}"
        group_data = self.redis.get(group_key)

        if not group_data:
            return None

        group_info = json.loads(group_data)

        # Ajouter la nouvelle image upscale
        new_image = {
            'url': image_url,
            'upscale_number': upscale_number,
            'timestamp': datetime.now().isoformat()
        }

        group_info['images'].append(new_image)

        # Mettre à jour le statut si toutes les images sont reçues
        if len(group_info['images']) == 4:
            group_info['status'] = 'complete'

        # Sauvegarder les modifications
        self.redis.setex(
            group_key,
            TEMP_STORAGE_DURATION,
            json.dumps(group_info)
        )

        return group_info


def cleanup_expired_images():
    """Nettoie les images expirées de Redis"""
    pattern = "temp:image:*"
    for key in redis_client.keys(pattern):
        if not redis_client.ttl(key):  # Si pas de TTL ou expiré
            redis_client.delete(key)


class MidjourneyGroupManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "midjourney"

    def create_or_update_group(self, task_id, prompt, initial_grid=None):
        """Creates a new group or updates an existing one"""
        group_key = f"{self.prefix}_group:{task_id}"

        # Check if group already exists
        existing_group = self.redis.get(group_key)
        if existing_group:
            try:
                group_data = json.loads(existing_group)
                # Update initial grid if provided and not already set
                if initial_grid and not group_data.get('initial_grid'):
                    group_data['initial_grid'] = initial_grid

                # Save back to Redis
                self.redis.setex(
                    group_key,
                    TEMP_STORAGE_DURATION,
                    json.dumps(group_data)
                )
                return group_data

            except json.JSONDecodeError:
                # If current data is corrupt, create new
                pass

        # Create new group
        group_data = {
            'task_id': task_id,
            'prompt': prompt,
            'initial_grid': initial_grid,
            'images': [],
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        # Store in Redis
        self.redis.setex(
            group_key,
            TEMP_STORAGE_DURATION,
            json.dumps(group_data)
        )

        app.logger.error(f"Created/updated Midjourney group: {task_id}")
        return group_data

    def add_image_to_group(self, task_id, image_data):
        """Adds an image to a group, updating status as needed"""
        group_key = f"{self.prefix}_group:{task_id}"
        group_data_bytes = self.redis.get(group_key)

        if not group_data_bytes:
            app.logger.error(f"Group {task_id} not found for adding image")
            return None

        try:
            group_data = json.loads(group_data_bytes)

            # Check if this variation already exists in the group
            variation_number = image_data.get('variation_number')
            existing_index = None

            for i, img in enumerate(group_data['images']):
                if img.get('variation_number') == variation_number:
                    existing_index = i
                    break

            # Add key field if not present
            if 'key' not in image_data:
                image_data['key'] = f"midjourney_image:{task_id}:{variation_number}"

            # Replace or append
            if existing_index is not None:
                group_data['images'][existing_index] = image_data
            else:
                group_data['images'].append(image_data)

            # Update status if we have all 4 images
            if len(group_data['images']) >= 4:
                group_data['status'] = 'completed'

                # Update the task metadata as well
                task_metadata_key = f"midjourney_task:{task_id}"
                self.redis.hset(task_metadata_key, 'status', b'completed')

                app.logger.error(f"Completed Midjourney group {task_id} with all 4 images")

            # Save back to Redis
            self.redis.setex(
                group_key,
                TEMP_STORAGE_DURATION,
                json.dumps(group_data)
            )

            # Also save the individual image
            image_key = image_data.get('key')
            if image_key:
                self.redis.setex(
                    image_key,
                    TEMP_STORAGE_DURATION,
                    json.dumps(image_data)
                )

            return group_data

        except json.JSONDecodeError as e:
            app.logger.error(f"Error parsing group data for {task_id}: {e}")
            return None

    def get_group(self, task_id):
        """Retrieves a group by task_id"""
        group_key = f"{self.prefix}_group:{task_id}"
        group_data_bytes = self.redis.get(group_key)

        if not group_data_bytes:
            return None

        try:
            return json.loads(group_data_bytes)
        except json.JSONDecodeError:
            app.logger.error(f"Error parsing group data for {task_id}")
            return None

    def get_image(self, task_id, variation_number):
        """Retrieves a specific image from a group"""
        image_key = f"midjourney_image:{task_id}:{variation_number}"
        image_data_bytes = self.redis.get(image_key)

        if not image_data_bytes:
            return None

        try:
            return json.loads(image_data_bytes)
        except json.JSONDecodeError:
            app.logger.error(f"Error parsing image data for {task_id}:{variation_number}")
            return None


image_manager = ImageManager(storage_manager)
midjourney_group_manager = MidjourneyGroupManager(storage_manager)
audio_manager = AudioManager(storage_manager)
video_manager = VideoManager(storage_manager)


class MidjourneyImageGroup:
    def __init__(self, task_id, prompt):
        self.task_id = task_id
        self.prompt = prompt
        self.images = []
        self.timestamp = datetime.now().isoformat()
        self.status = 'pending'  # pending, partial, complete

    def add_image(self, image_data):
        """Ajoute une image au groupe"""
        self.images.append(image_data)
        self._update_status()

    def _update_status(self):
        """Met à jour le statut en fonction du nombre d'images"""
        if len(self.images) == 4:
            self.status = 'complete'
        elif len(self.images) > 0:
            self.status = 'partial'
        else:
            self.status = 'pending'

    def to_dict(self):
        """Convertit le groupe en dictionnaire"""
        return {
            'task_id': self.task_id,
            'prompt': self.prompt,
            'images': self.images,
            'timestamp': self.timestamp,
            'status': self.status
        }


class ChatManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.image_manager = ImageManager(redis_client)
        self.prefix = "chat"

    def store_message(self, user_id, message_data):
        """
        Stocke un message
        """
        message_id = f"{self.prefix}:{user_id}:{datetime.now().timestamp()}"

        timestamp_key = f"app:chat:timestamps:{user_id}"
        current_timestamp = datetime.now().timestamp()
        self.redis.zadd(timestamp_key, {message_id: current_timestamp})


        #prépare les donnés du message
        data = {
            'user_id': user_id,
            'text': message_data.get('text',  ''),
            'timestamp': datetime.now().isoformat(),
            'type': message_data.get('type', 'text'),  # text, image, enhanced_image
            'message_id': message_id
        }
        # Gère les images attachées
        if message_data.get('image_data'):
            image_key = self.image_manager.store_temp_image(
                user_id,
                message_data['image_data'],
                {'type': 'chat_image'}
            )
            data['image_key'] = image_key

        # Utilise un pipeline pour les opérations atomiques
        pipe = self.redis.pipeline()

        # Stocke le message
        pipe.hmset(message_id, data)
        pipe.expire(message_id, PERMANENT_STORAGE_DURATION)

        # Ajoute à l'historique utilisateur
        pipe.lpush(f"user:{user_id}:messages", message_id)

        # Expire l'index des timestamps aussi
        pipe.expire(timestamp_key, PERMANENT_STORAGE_DURATION)

        # Exécute toutes les opérations
        pipe.execute()

        return message_id

    def get_messages_by_timerange(self, user_id, start_time, end_time):
        """
        Récupère les messages dans une plage temporelle
        """
        timestamp_key = f"app:chat:timestamps:{user_id}"
        message_ids = self.redis.zrangebyscore(timestamp_key, start_time, end_time)

        if not message_ids:
            return []

        # Utilise un pipeline pour récupérer tous les messages
        pipe = self.redis.pipeline()
        for mid in message_ids:
            pipe.hgetall(mid)

        messages = pipe.execute()

        # Traite les messages
        return [
            {k.decode('utf-8'): v.decode('utf-8') for k, v in msg.items()}
            for msg in messages if msg
        ]

    def get_message(self, message_id):
        """Récupère un message spécifique"""
        message = self.redis.hgetall(message_id)
        if message and message.get('image_key'):
            message['image_url'] = f"/image/{message['image_key']}"
        return message

    def get_user_chat_history(self, user_id, page=1, per_page=20, filter_type=None):
        """
        Récupère l'historique des messages avec filtrage optionnel
        """


        # Clé pour l'index des messages
        user_messages_key = f"user:{user_id}:messages"

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page - 1

        # Récupère les IDs des messages avec filtrage potentiel
        message_ids = self.redis.lrange(user_messages_key, start_idx, end_idx)

        if not message_ids:
            return {
                'messages': [],
                'page': page,
                'has_more': False
            }

        # Utilise un pipeline pour récupérer tous les messages et leurs métadonnées en une seule opération
        pipeline = self.redis.pipeline()
        for mid in message_ids:
            pipeline.hgetall(mid)

        # Exécute toutes les commandes en une fois
        results = pipeline.execute()

        # Récupère les messages avec leurs images
        messages = []
        for msg in results:
            if filter_type and msg.get('type') != filter_type:
                continue

            if msg.get('image_key'):
                msg['image_url'] = f"/image/{msg['image_key']}"

            # Décode les valeurs bytes en strings si nécessaire
            processed_msg = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in msg.items()
            } if msg else {}

            if processed_msg:  # N'ajoute que les messages non vides
                messages.append(processed_msg)

        # Vérifie s'il y a plus de messages
        next_page_check = self.redis.lrange(user_messages_key, end_idx + 1, end_idx + 1)

        return {
            'messages': messages,
            'page': page,
            'has_more': bool(next_page_check)
        }

    def search_message(self, user_id, query, page=1, per_page=20):
        """
                Recherche dans l'historique des messages
                """
        # Implémentation dépendrait de la mise en place d'un index de recherche (PLUS TARD !)

# Initialisation
chat_manager = ChatManager(redis_client)

def get_paginated_history(wp_user_id, history_type):
    """
    Fonction utilitaire pour gérer la pagination de l'historique
    """
    try:
        # Récupère le numéro de page et le nombre d'éléments par page depuis l'URL
        page = request.args.get('page', 1, type=int)
        per_page = min(50, request.args.get('per_page', 20, type=int))

        # Log des informations
        app.logger.error(f"Fetching {history_type} history for user {wp_user_id}")
        app.logger.error(f"Page: {page}, Items per page: {per_page}")

        # Récupère l'historique complet pour le type demandé
        history = image_manager.get_user_history(
            user_id=wp_user_id,
            history_type=history_type
        )

        # Calcule les indices pour la pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Extrait la portion demandée de l'historique
        items = history[start_idx:end_idx] if history else []
        total_items = len(history) if history else 0

        # Retourne la réponse formatée
        return {
            'success': True,
            'data': {
                'items': items,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': total_items,
                    'has_more': total_items > end_idx
                }
            }
        }, 200

    except Exception as e:
        app.logger.error(f"Error fetching {history_type} history: {str(e)}")
        return {
            'success': False,
            'error': f'Failed to fetch {history_type} history'
        }, 500


def initialize_user_tokens(wp_user_id, subscription_level=None, force_reset=False):
    """
    Initialiser ou recharger les tokens d'un utilisateur
    en tenant compte de son historique de consommation

    Args:
        wp_user_id: ID de l'utilisateur
        subscription_level: Niveau d'abonnement
        force_reset: Si True, force la réinitialisation des tokens quoi qu'il arrive
    """
    try:
        app.logger.error(f"Initializing tokens for user {wp_user_id} with subscription {subscription_level}")

        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        # Vérifier si l'utilisateur existe déjà dans la table user_tokens
        cursor.execute("SELECT id, tokens_remaining FROM user_tokens WHERE wp_user_id = %s", (wp_user_id,))
        user_record = cursor.fetchone()

        app.logger.error(f"[TOKENS] Current user record: {user_record}")

        # Récupérer les informations sur l'abonnement pour calculer la période de rechargement
        client = get_client_by_wp_user_id(wp_user_id)
        if client:
            status = client.get("status")
            # Pour 'canceled', vérifier si dans la période de grâce
            if status == 'canceled':
                expiration_date = client.get("expiration_date")
                if expiration_date:
                    grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
                    if datetime.now() > grace_end:
                        # En dehors de la période de grâce, retirer tous les tokens
                        cursor.execute("""
                                    UPDATE user_tokens
                                    SET tokens_remaining = 0
                                    WHERE wp_user_id = %s
                                """, (wp_user_id,))
                        conn.commit()
                        app.logger.error(f"Période de grâce expirée pour l'utilisateur {wp_user_id}, tokens mis à 0")
                        cursor.close()
                        conn.close()
                        return 0

        # Déterminer le nombre de tokens en fonction du niveau d'abonnement
        max_tokens = get_tokens_for_subscription(subscription_level)
        app.logger.error(f"[TOKENS] Max tokens calculated: {max_tokens} for subscription {subscription_level}")
        now = datetime.now()

        # Convertir en int si c'est une chaîne numérique
        if isinstance(subscription_level, str) and subscription_level.isdigit():
            subscription_level = int(subscription_level)

        # Déterminer si c'est un plan gratuit et la période de rechargement
        is_free_plan = is_free_subscription_plan(subscription_level)
        app.logger.error(f"[TOKENS] Is free plan: {is_free_plan}")
        # Période de rechargement selon le type d'abonnement
        refill_days = REFILL_PERIODS[
            SUBSCRIPTION_TYPES["BASIC_FREE"] if is_free_plan else SUBSCRIPTION_TYPES["BASIC_PAID"]]
        next_refill = now + timedelta(days=refill_days)
        app.logger.error(f"[TOKENS] Next refill: {next_refill} (in {refill_days} days)")

        if user_record:
            # Si force_reset est True ou si c'est la période de rechargement, réinitialiser à max_tokens
            if force_reset:
                tokens_to_set = max_tokens
                app.logger.error(
                    f"Forçage de réinitialisation des tokens pour l'utilisateur {wp_user_id} à {max_tokens}")
            else:
                tokens_to_set = user_record[1]  # Garder le nombre actuel

            cursor.execute("""
                        UPDATE user_tokens 
                        SET tokens_remaining = %s, 
                            last_refill = %s,
                            next_refill = %s
                        WHERE wp_user_id = %s
                    """, (tokens_to_set, now, next_refill, wp_user_id))
        else:
            # Création d'un nouvel enregistrement
            tokens_to_set = max_tokens

            cursor.execute("""
                        INSERT INTO user_tokens (wp_user_id, tokens_remaining, last_refill, next_refill)
                        VALUES (%s, %s, %s, %s)
                    """, (wp_user_id, tokens_to_set, now, next_refill))

        conn.commit()
        app.logger.error(
            f"Tokens initialized for user {wp_user_id}: {tokens_to_set} tokens, next refill in {refill_days} days")
        return tokens_to_set

    except Exception as e:
        app.logger.error(f"Error initializing tokens: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def get_tokens_for_subscription(subscription_level):
    """
    Définir le nombre de tokens en fonction du niveau d'abonnement
    """

    # Si le niveau d'abonnement est numérique (identifiant WordPress)
    if isinstance(subscription_level, str) and subscription_level.isdigit():
        subscription_level = int(subscription_level)

    # Vérification explicite pour les IDs d'abonnement connus
    if subscription_level == 28974 or subscription_level == "28974":
        return TOKEN_ALLOCATION[SUBSCRIPTION_TYPES["BASIC_PAID"]]  # 500 tokens
    elif subscription_level == 29094 or subscription_level == "29094":
        return TOKEN_ALLOCATION[SUBSCRIPTION_TYPES["BASIC_FREE"]]  # 150 tokens

    mapped_level = SUBSCRIPTION_ID_MAPPING.get(subscription_level, SUBSCRIPTION_TYPES["BASIC_FREE"])

    if mapped_level in TOKEN_ALLOCATION:
        return TOKEN_ALLOCATION[mapped_level]

    app.logger.error(f"Subscription level unknown {subscription_level}, basic plan set by default for user token")
    return TOKEN_ALLOCATION[SUBSCRIPTION_TYPES["BASIC_FREE"]]  # 150 par défaut = BASIC


def get_user_tokens(wp_user_id):
    """
    Récupérer le nombre de tokens restants pour un utilisateur
    et gérer le rechargement automatique si nécessaire
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT tokens_remaining, next_refill, last_refill
            FROM user_tokens
            WHERE wp_user_id = %s
        """, (wp_user_id,))

        result = cursor.fetchone()

        if result:
            tokens_remaining, next_refill, last_refill = result

            # Vérifier l'état de l'abonnement
            client = get_client_by_wp_user_id(wp_user_id)
            if client:
                status = client.get("status")

                # Si l'abonnement est 'canceled', vérifier si dans la période de grâce
                if status == 'canceled':
                    expiration_date = client.get("expiration_date")
                    if expiration_date:
                        grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
                        if datetime.now() > grace_end:
                            # En dehors de la période de grâce, retirer tous les tokens
                            cursor.execute("""
                                            UPDATE user_tokens
                                            SET tokens_remaining = 0
                                            WHERE wp_user_id = %s
                                        """, (wp_user_id,))
                            conn.commit()
                            app.logger.error(f"Période de grâce expirée pour l'utilisateur {wp_user_id}, tokens mis à 0")
                            return 0

                # Si abonnement abandonné ou expiré, pas de tokens
                if status in ['abandoned', 'expired']:
                    return 0

            # Vérifier si un rechargement est nécessaire
            now = datetime.now()
            if next_refill and next_refill <= now:
                app.logger.error(f"Automatic token refill triggered for user {wp_user_id}")

                # Récupérer le niveau d'abonnement actuel
                subscription_level = client.get("subscription_level") if client else None

                # Réinitialiser complètement les tokens à la valeur maximale
                max_tokens = get_tokens_for_subscription(subscription_level)

                # Calculer la prochaine date de rechargement
                is_free = is_free_subscription_plan(subscription_level)
                refill_days = REFILL_PERIODS[
                    SUBSCRIPTION_TYPES["BASIC_FREE"] if is_free else SUBSCRIPTION_TYPES["BASIC_PAID"]]
                new_next_refill = now + timedelta(days=refill_days)

                cursor.execute("""
                                UPDATE user_tokens
                                SET tokens_remaining = %s,
                                    last_refill = %s,
                                    next_refill = %s
                                WHERE wp_user_id = %s
                            """, (max_tokens, now, new_next_refill, wp_user_id))

                conn.commit()
                tokens_remaining = max_tokens
                app.logger.error(
                    f"Tokens refilled for user {wp_user_id}: {tokens_remaining}, next refill in {refill_days} days")

            return tokens_remaining

            # Si l'utilisateur n'existe pas, initialiser ses tokens
        client = get_client_by_wp_user_id(wp_user_id)
        subscription_level = client.get("subscription_level") if client else None
        tokens = initialize_user_tokens(wp_user_id, subscription_level)
        return tokens

    except Exception as e:
        app.logger.error(f"Error getting user tokens: {str(e)}")
        return 0
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def calculate_next_refill_date(subscription_level):
    """
        Calcule la prochaine date de rechargement en fonction du niveau d'abonnement
        """
    now = datetime.now()

    if is_free_subscription_plan(subscription_level):
        return now + timedelta(days=3)  # Tous les 3 jours pour plans gratuits
    else:
        return now + timedelta(days=30)  # Tous les 30 jours pour plans payants

def update_all_users_tokens():
    """
    Met à jour les tokens de tous les utilisateurs en fonction de leur niveau d'abonnement actuel
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        # Récupérer tous les utilisateurs
        cursor.execute("SELECT wp_user_id, subscription_level FROM clients")
        users = cursor.fetchall()

        updated_count = 0
        for user_id, subscription_level in users:
            # Déterminer le nombre de tokens approprié
            tokens = get_tokens_for_subscription(subscription_level)
            now = datetime.now()

            # Vérifier si c'est un plan gratuit pour la période de rechargement
            is_free_plan = (subscription_level == 29094 or (
                        isinstance(subscription_level, str) and subscription_level == "29094"))
            next_refill = now + timedelta(days=3 if is_free_plan else 30)

            # Mettre à jour les tokens
            cursor.execute("""
                UPDATE user_tokens
                SET tokens_remaining = %s,
                    last_refill = %s,
                    next_refill = %s
                WHERE wp_user_id = %s
            """, (tokens, now, next_refill, user_id))

            if cursor.rowcount > 0:
                updated_count += 1
                app.logger.error(f"Updated tokens for user {user_id} to {tokens}")

        conn.commit()
        app.logger.info(f"Updated tokens for {updated_count} users")
        return updated_count

    except Exception as e:
        app.logger.error(f"Error updating tokens: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()
def use_tokens(wp_user_id, amount):
    """
    Utiliser des tokens pour une opération
    Retourne True si les tokens ont été consommés avec succès, False sinon
    """
    try:
        # Vérifier si l'utilisateur a assez de tokens
        tokens_remaining = get_user_tokens(wp_user_id)

        app.logger.error(f"Token usage request: User {wp_user_id} remaining: {tokens_remaining} is attempting tot use {amount} token (has {tokens_remaining})")
        if tokens_remaining < amount:
            app.logger.warning(f"User {wp_user_id} has insufficient tokens: {tokens_remaining} < {amount}")
            return False

        # Diminuer les tokens
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        # IMPORTANT: Ajouter une transaction pour éviter les problèmes de concurrence
        cursor.execute("BEGIN")

        cursor.execute("""
            SELECT tokens_remaining FROM user_tokens
            WHERE wp_user_id = %s
            FOR UPDATE
        """, (wp_user_id,))

        current_tokens = cursor.fetchone()[0]

        # Vérifier à nouveau (en cas de concurrence)
        if current_tokens < amount:
            cursor.execute("ROLLBACK")
            app.logger.warning(f"Race condition detected: User {wp_user_id} now has {current_tokens} tokens")
            return False

        # Mise à jour des tokens
        cursor.execute("""
                UPDATE user_tokens
                SET tokens_remaining = tokens_remaining - %s
                WHERE wp_user_id = %s
            """, (amount, wp_user_id))

        cursor.execute("COMMIT")

        # LOG détaillé après la transaction
        app.logger.error(f"Tokens used: User {wp_user_id} used {amount} tokens. New blance: {tokens_remaining- amount}")
        return True

    except Exception as e:
        app.logger.error(f"Error using tokens: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def is_free_subscription_plan(subscription_level):
    """
    Détermine si un abonnement correspond à un plan gratuit
    IMPORTANT: Seul 29094 est le vrai plan gratuit.
    """
    # Conversion en int si c'est une chaîne numérique
    if isinstance(subscription_level, str) and subscription_level.isdigit():
        subscription_level = int(subscription_level)

    # Vérification directe avec le mapping
    mapped_level = SUBSCRIPTION_ID_MAPPING.get(subscription_level)
    if mapped_level == SUBSCRIPTION_TYPES["BASIC_FREE"]:
        return True

    # Liste des IDs uniquement pour les véritables plans gratuits
    free_plan_id = 29094
    return (
        subscription_level == free_plan_id or
        (isinstance(subscription_level, str) and subscription_level == "29094") or
        (subscription_level == SUBSCRIPTION_TYPES["BASIC_FREE"])
    )

def is_in_grace_period(client):
    """Détermine si un client est dans sa période de grâce"""
    if client.get("status") == "grace_period":
        return True
    if client.get("status") == "canceled":
        expiration_date = client.get("expiration_date")
        if expiration_date:
            grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
            return datetime.now() <= grace_end
    return False


def update_grace_period_status():
    """Met à jour le statut des utilisateurs dont la période de grâce expire"""
    conn = psycopg2.connect(...)
    cursor = conn.cursor()

    # Identifie les utilisateurs dont la période de grâce a expiré
    cursor.execute("""
        SELECT id FROM clients 
        WHERE status = 'grace_period' 
        AND expiration_date + interval '%s days' < NOW()
    """, (GRACE_PERIOD,))

    expired_clients = cursor.fetchall()

    # Pour chaque client identifié, mettre à jour le statut
    for client_id, in expired_clients:
        cursor.execute("""
            UPDATE clients 
            SET status = 'expired' 
            WHERE id = %s
        """, (client_id,))

        # Mettre à jour les tokens et permissions
        revoke_client_permissions(client_id)
        # Mettre les tokens à zéro
        # ...

    conn.commit()
    cursor.close()
    conn.close()

def refund_tokens(wp_user_id, amount):
    """
    Rembourser des tokens en cas d'échec de l'opération
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE user_tokens
            SET tokens_remaining = tokens_remaining + %s
            WHERE wp_user_id = %s
        """, (amount, wp_user_id))

        conn.commit()
        app.logger.info(f"Refunded {amount} tokens to user {wp_user_id}")
    except Exception as e:
        app.logger.error(f"Error refunding tokens: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
def get_client_by_wp_user_id(wp_user_id):
    app.logger.error(f"Attempting to get client with wp_user_id: {wp_user_id}")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    app.logger.error(f"Executing client lookup query for wp_user_id: {wp_user_id}")
    cursor.execute("""
        SELECT id, email, subscription_level, status, expiration_date FROM clients WHERE wp_user_id = %s
    """, (wp_user_id,))
    client = cursor.fetchone()
    cursor.close()
    conn.close()
    if client:
        result = {
            "id": client[0],
            "email": client[1],
            "subscription_level": client[2],
            "status": client[3],
            "expiration_date": client[4],
        }
        app.logger.error(f"Found client data: {result}")
        return result
    app.logger.error("No client found in Client table")
    return None

def add_permission(client_id, action):
    app.logger.error(f"Adding permission - Client ID: {client_id}, Action: {action}")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO permissions (client_id, action, is_allowed)
        VALUES (%s, %s, TRUE)
    """, (client_id, action))
    conn.commit()
    cursor.close()
    conn.close()
    app.logger.error("Permission added successfully")

def check_client_permission(client_id, action):
    app.logger.error(f"Checking permission for client {client_id}, action: {action}")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.is_allowed, c.expiration_date, c.status, c.subscription_level
        FROM permissions AS p
        JOIN clients AS c ON p.client_id = c.id
        WHERE p.client_id = %s AND p.action = %s
    """, (client_id, action))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        is_allowed, expiration_date, status, subscription_level = result
        now = datetime.now()

        is_free_plan = is_free_subscription_plan(subscription_level)

        # Add debug logs for timezone investigation
        app.logger.error(f"""
        Timezone Debug:
        Current server time (now): {now}
        Expiration date from DB: {expiration_date}
        Status: {status}
        Is allowed: {is_allowed}
        Subscription level: {subscription_level}
        Is free plan: {is_free_plan}
        """)

        is_in_grace = False
        if status == 'grace_period':
            is_in_grace = True
            app.logger.error(f"Client with explicit grace_period status")
        elif status == 'canceled' and not is_free_plan: # Grace period only applies to paid plans
            grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
            is_in_grace = now <= grace_end
            app.logger.error(
                f"Client in grace period calculation: now={now}, grace_end={grace_end}, is_in_grace={is_in_grace}")


        #régles d'accès unifiées
        if is_allowed:
            # 1. Plans gratuits uniquement si 'active'
            if is_free_plan:
                if status == 'active':
                    app.logger.error(f"Free plan permission check passed - Is Allowed: {is_allowed}, Status: {status}")
                    return True
            # 2. Pour les plans payants, on vérifie la date d'expiration
            else:
                # abonnement actif et non expiré
                if status == 'active' and now <= expiration_date:
                    app.logger.error(f"Paid plan permission check passed - Is Allowed: {is_allowed}, Status: {status}")
                    return True
                # En période de grâce explicite ou caluclée
                elif is_in_grace:
                    app.logger.error(f"Plan in grace period permission check passed")
                    return True

        app.logger.error("Permission check failed")
        return False

    app.logger.error("No permission record found")
    return False

def verify_jwt_token(token):
    try:
        # Décodage du token, s'assurant que le résultat est un dictionnaire
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return decoded_token  # Retourne le dictionnaire JSON décodé ENTIER !
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        app.logger.error("=== Decorator Start ===")
        app.logger.error(f"Function being decorated: {f.__name__}")
        app.logger.error(f"Args: {args}")
        app.logger.error(f"Kwargs: {kwargs}")
        app.logger.error("=== Token Extraction Debug ===")
        app.logger.error(f"Full URL received: {request.url}")
        app.logger.error(f"URL Path: {request.path}")
        app.logger.error(f"Query Parameters: {request.args}")

        token = request.args.get('token')  # Retrieve token from URL query parameter
        app.logger.error(" Checking token from URL parameters")
        app.logger.error(f"Token extracted: {token}")


        if not token:
            app.logger.error("No token found in request")
            auth_header = request.headers.get('Authorization')  # Fallback to Authorization header
            if auth_header and auth_header.startswith("Bearer "):
                app.logger.error(f"Authorization header: {auth_header}")
                token = auth_header.split(" ")[1]
                app.logger.error(f"Token extracted from header: {token}")

        if not token:
            app.logger.error("No token found in either URL or headers-redirection-")
            return handle_error_response(
                "Please log in to continue",
                401,
                "Authentication Required"
            )

        if token == FIXED_TOKEN:
            return f(None, **kwargs)
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            app.logger.error(f"Token decoded successfully: {decoded_token}")
            wp_user_id = decoded_token.get("data", {}).get("user", {}).get("id")
            app.logger.error(f"WP User ID extracted: {wp_user_id}")
            if not wp_user_id:
                app.logger.error("No wp_user_id found in token --redirection--")
                return redirect(LOGIN_URL)

            client = get_client_by_wp_user_id(wp_user_id)
            app.logger.error(f"Client lookup result: {client}")
            if not client:
                return redirect(LOGIN_URL)

            return f(wp_user_id, **kwargs)

        except jwt.ExpiredSignatureError:
            app.logger.error("Token has expired")
            return handle_error_response(
                "Your session has expired. Please log in again.",
                401,
                "Session Expired"
            )
        except jwt.InvalidTokenError as e:
            app.logger.error(f"Token decode error:{str(e)} ")
            return handle_error_response(
                "Invalid authentication token",
                403,
                "Invalid Token"
            )
            #Next Step : mise en place redirection template error html
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}")
            return abort(500, description="Internal server error")

    return decorated_function

def add_client(wp_user_id, email, subscription_level, status, expiration_date):
    app.logger.error(f"Adding new client - WP User ID: {wp_user_id}, Email: {email}, Level: {subscription_level}")
    start_date = datetime.now()
    expiration_date = start_date + timedelta(days=1)

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    app.logger.error("Executing client insert query")
    cursor.execute("""
        INSERT INTO clients (wp_user_id, email, subscription_level, status, start_date, expiration_date)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """, (wp_user_id, email, subscription_level, status, datetime.now(), expiration_date))
    client_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    app.logger.error(f"Successfully added client with ID: {client_id}")
    return client_id

def update_client_subscription(client_id, subscription_level, status, expiration_date):
    app.logger.error(f"Updating subscription - Client ID: {client_id}, Level: {subscription_level}, Status: {status}")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    # Récupère la date de début actuelle
    cursor.execute("""
        UPDATE clients SET subscription_level = %s, status = %s, expiration_date = %s WHERE id = %s
    """, (subscription_level, status, expiration_date, client_id))
    rows_affected = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    app.logger.error(f"Subscription updated. Rows affected: {rows_affected}")


def assign_permissions(client_id, subscription_level):
    """
        Attribue les permissions nécessaires à un client sans supprimer celles qui existent déjà
    """
    # Convert subscription_level to int if it's numeric
    if isinstance(subscription_level, str) and subscription_level.isdigit():
        subscription_level = int(subscription_level)

    mapped_level = SUBSCRIPTION_ID_MAPPING.get(subscription_level, subscription_level)
    if isinstance(mapped_level, str) and mapped_level not in PERMISSIONS:
        # Si le niveau mappé n'est pas reconnu, utiliser basic par défaut
        mapped_level = SUBSCRIPTION_TYPES["BASIC_FREE"]

    # Map the subscription level to our internal levels
    app.logger.error(f"Mapping subscription {subscription_level} to {mapped_level}")

    required_permissions = set(PERMISSIONS.get(mapped_level, PERMISSIONS[SUBSCRIPTION_TYPES["BASIC_FREE"]]))

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()

    # Récupérer les permissions actuelles
    cursor.execute("SELECT action FROM permissions WHERE client_id = %s", (client_id,))
    current_permissions = {row[0] for row in cursor.fetchall()}

    permissions_to_add = required_permissions - current_permissions

    permissions_to_remove = current_permissions - required_permissions

    for action in permissions_to_add:
        cursor.execute("""
                    INSERT INTO permissions (client_id, action, is_allowed)
                    VALUES (%s, %s, TRUE)
                """, (client_id, action))
        app.logger.error(f"Permission '{action}' ajoutée au client ID {client_id}")

    # Supprimer les permissions en trop (optionnel selon votre logique métier)
    # Pour certains scénarios, vous pourriez vouloir conserver les permissions supplémentaires
    for action in permissions_to_remove:
        cursor.execute("""
            DELETE FROM permissions 
            WHERE client_id = %s AND action = %s
        """, (client_id, action))
        app.logger.error(f"Permission '{action}' supprimée du client ID {client_id}")

    conn.commit()
    cursor.close()
    conn.close()
    return len(permissions_to_add), len(permissions_to_remove)

def check_and_revoke_permissions():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    # Vérifie les clients annulés dont l'expiration_date est dépassée
    cursor.execute("""
        SELECT id, subscription_level FROM clients WHERE (status = 'canceled' OR status = 'expired' OR status = 'abandonned') AND expiration_date <= %s
    """, (datetime.now(),))
    clients_to_check = cursor.fetchall()

    for client_row in clients_to_check:
        client_id, subscription_level = client_row

        # Vérifier si c'est un plan gratuit (qui ne devrait pas être révoqué)
        is_free_plan = (
                str(subscription_level) == "29094" or
                (isinstance(subscription_level, str) and subscription_level.isdigit() and int(
                    subscription_level) == 29094) or
                str(subscription_level) == "28974" or
                (isinstance(subscription_level, str) and subscription_level.isdigit() and int(
                    subscription_level) == 28974)
        )

        if not is_free_plan:
            # Révoquer les permissions uniquement pour les plans payants
            revoke_client_permissions(client_id)
            app.logger.error(f"Permissions revoked for client ID {client_id} (plan: {subscription_level}) due to expired subscription.")

    conn.commit()
    cursor.close()
    conn.close()

def revoke_client_permissions(client_id):
    app.logger.error(f"Revoking all permissions for client ID: {client_id}")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("DELETE FROM permissions WHERE client_id = %s", (client_id,))
    rows_deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    app.logger.error(f"Permissions revoked. Total permissions deleted: {rows_deleted}")

def update_client_expiry(client_id, days=1):
    """
        Met à jour la date d'expiration pour un abonnement en ajoutant 'days' jours
        si le statut est encore actif.
    """
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()

    # Récupère la date d'expiration actuelle
    cursor.execute("SELECT expiration_date FROM clients WHERE id = %s", (client_id,))
    result = cursor.fetchone()[0]
    current_expiration_date, status = result if result else (None, None)[0]

    if status != 'active':
        # Si le statut n'est pas actif, ne prolonge pas l'abonnement
        app.logger.info(f"Client ID {client_id} n'est pas actif, pas de mise à jour de la date d'expiration.")
        cursor.close()
        conn.close()
        return

    # prolonge l'abonnement si le client est actif
    new_expiration_date = (current_expiration_date if current_expiration_date and current_expiration_date > datetime.now() else datetime.now()) + timedelta(days=days)

    cursor.execute("""
        UPDATE clients SET expiration_date = %s WHERE id = %s
    """, (new_expiration_date, client_id))
    conn.commit()
    cursor.close()
    conn.close()


def handle_error_response(message, status_code, title="Error"):
    """Enhanced error handler with template support"""
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get(
            'Accept') == 'application/json':
        return jsonify({
            "error": "authentication_error",
            "message": message,
            "redirect_url": LOGIN_URL
        }), status_code

    # For regular requests, redirect to error page with parameters
    error_url = f'/error?title={quote(title)}&message={quote(message)}'
    return redirect(error_url)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_format(img_format):
    supported_formats = ['PNG', 'JPEG', 'JPG', 'GIF']
    if img_format not in supported_formats:
        return False, f"Unsupported format: {img_format}"
    if img_format == 'JPG':
        img_format = 'JPEG'
        #img_format = 'PDF'
        #img_format = 'webp'
    return True, img_format


@app.route('/')
@token_required
def index(wp_user_id):
    tokens_remaining = get_user_tokens(wp_user_id)
    client = get_client_by_wp_user_id(wp_user_id)

    return render_template('index.html', tokens_remaining=tokens_remaining, client=client)

@app.route('/refresh-token')
def refresh_token():
    """Endpoint for handling expired tokens"""
    return jsonify({
        "error": "token_expired",
        "message": "Your session has expired. Please log in again.",
        "redirect_url": LOGIN_URL
    }), 401

@app.route('/error')
def error_page():
    title = request.args.get('title', 'Error')
    message = request.args.get('message', 'An error occurred')
    return render_template('error-message.html', title=title, message=message, LOGIN_URL=LOGIN_URL)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({
        'eror': 'File too large',
        'message': 'Le fichier ne doit pas dépaseer 20 MB'
    }), 413

@app.route('/sync-membership', methods=['POST'])
@token_required
def sync_membership(wp_user_id_from_token):
    #Si le token est fixe, `wp_user_id` sera `None`
    if wp_user_id_from_token is None:
        auth_header = request.headers.get('Authorization')
        if auth_header != f'Bearer {FIXED_TOKEN}':
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    app.logger.error(f"Received subscription sync data: {data}")

    required_fields = ['wp_user_id', 'email', 'subscription_level', 'status', 'start_date', 'expiration_date']
    if not all(field in data for field in required_fields):
        missing_fields = [field for field in required_fields if field not in data]
        app.logger.error(f"Missing required fields: {missing_fields}")
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

    try:
        # Convert wp_user_id to integer
        wp_user_id = int(data['wp_user_id'])
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d %H:%M:%S')

        subscription_level = data['subscription_level']
        if isinstance(subscription_level, str) and subscription_level.isdigit():
            subscription_level = int(subscription_level)

        # Vérifier si c'est le plan gratuit
        is_free_plan = is_free_subscription_plan(subscription_level)

        if is_free_plan:
            # Pour les plans gratuits
            expiration_date = datetime.now() + timedelta(days=365)  # Date d'expiration lointaine

            # Préserver le statut 'abandoned' et ne pas le forcer à 'active'
            if data['status'] != 'abandoned':
                data['status'] = 'active'

            app.logger.error(f"Plan gratuit détecté pour l'utilisateur ID {wp_user_id}, expiration définie à 1 an")
        else:
            # Pour les plans payants
            expiration_date = datetime.strptime(data['expiration_date'], '%Y-%m-%d %H:%M:%S')

            # Check if expiration date has passed
            if expiration_date < datetime.now():
                if data['status'] == 'canceled':
                    # Vérifier si la période de grâce est aussi dépassée
                    grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
                    if datetime.now() > grace_end:
                        data['status'] = 'expired'
                        app.logger.error(f"Abonnement marqué comme 'expired' - période de grâce dépassée")
                else:
                    data['status'] = 'expired'
                    app.logger.error(f"Abonnement marqué comme 'expired' - date d'expiration dépassée")

        # Vérifie si le client existe
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.error(f"Looking up client with wp_user_id: {wp_user_id}")

        old_subscription = None
        old_status = None

        if client:
            old_subscription = client.get("subscription_level")
            old_status = client.get("status")
            client_id = client["id"]

            # Mise à jour de l'abonnement existant
            app.logger.error(f"Updating existing client with id: {client_id}, status: {data['status']}")
            update_client_subscription(
                client_id=client_id,
                subscription_level=data['subscription_level'],
                status=data['status'],
                expiration_date=expiration_date,
            )
            if data['status'] == 'active':
                app.logger.error(f"Attribution automatique des permissions pour client actif: {client_id}")
                added, removed = assign_permissions(client_id, data['subscription_level'])
                app.logger.error(f"Permissions mises à jour: {added} ajoutées, {removed} supprimées")
        else:
            # Création d'un nouveau client
            app.logger.error(f"Adding new client with wp_user_id: {wp_user_id}")
            client_id = add_client(
                wp_user_id=wp_user_id,
                email=data['email'],
                subscription_level=data['subscription_level'],
                status=data['status'],
                expiration_date=expiration_date,
            )

            # Gestion des transitions d'abonnement et des tokens
            # ----------------------------------------------

            # 1. Transition d'un plan payant à un plan gratuit
        if old_subscription and old_subscription != subscription_level:
            if (not is_free_plan and is_free_subscription_plan(old_subscription)):
                # Promotion: Plan gratuit -> Plan payant (réinitialiser à 500)
                initialize_user_tokens(wp_user_id, subscription_level, force_reset=True)
                app.logger.error(
                    f"Promotion vers plan payant: tokens réinitialisés à 500 pour l'utilisateur {wp_user_id}")
            elif (is_free_plan and not is_free_subscription_plan(old_subscription)):
                # Rétrogradation: Plan payant -> Plan gratuit (réduire à 150)
                conn = psycopg2.connect(
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                )
                cursor = conn.cursor()
                cursor.execute("""
                            UPDATE user_tokens
                            SET tokens_remaining = %s
                            WHERE wp_user_id = %s
                        """, (TOKEN_ALLOCATION[SUBSCRIPTION_TYPES["BASIC_FREE"]], wp_user_id))
                conn.commit()
                cursor.close()
                conn.close()
                app.logger.error(
                    f"Rétrogradation vers plan gratuit: tokens réduits à 150 pour l'utilisateur {wp_user_id}")
            else:
                # Pas de changement de type de plan, juste initialiser les tokens normalement
                initialize_user_tokens(wp_user_id, subscription_level)
        else:
            # Nouveau client ou pas de changement de plan
            initialize_user_tokens(wp_user_id, subscription_level)

            # 2. Transition vers un statut 'abandoned' ou 'expired'
        if data['status'] in ['abandoned', 'expired']:
            # Mettre les tokens à zéro immédiatement sauf pour les plans gratuits --doivent récupérer leurs nb de tokens--
            if not is_free_plan:
                conn = psycopg2.connect(
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                )
                cursor = conn.cursor()
                cursor.execute("""
                            UPDATE user_tokens
                            SET tokens_remaining = 0
                            WHERE wp_user_id = %s
                        """, (wp_user_id,))
                conn.commit()
                cursor.close()
                conn.close()
                app.logger.error(f"Statut {data['status']} (plan payant): tokens mis à zéro pour l'utilisateur {wp_user_id}")
            else:
                app.logger.error(
                    f"Statut {data['status']} (plan gratuit): conservation des tokens pour l'utilisateur {wp_user_id}")

            # Révoquer les permissions
            revoke_client_permissions(client_id)
            app.logger.error(f"Permissions révoquées pour l'utilisateur {wp_user_id} (statut: {data['status']})")

        elif data['status'] == 'canceled':
            # Vérifier si dans la période de grâce
            grace_end = expiration_date + timedelta(days=GRACE_PERIOD)
            if datetime.now() <= grace_end:
                data['status'] = 'grace_period' #nouveau statu
                # Pendant la période de grâce, maintenir les permissions
                app.logger.error(f"Plan annulé en période de grâce: maintien des permissions pour client_id: {client_id}")
                added, removed = assign_permissions(client_id, data['subscription_level'])
                app.logger.error(f"Permissions updated: {added} added, {removed} removed")
            else:
                app.logger.error(f"Plan annulé hors période de grâce: révocation des permissions pour client_id: {client_id}")
                revoke_client_permissions(client_id)
                # Mettre également les tokens à zéro
                conn = psycopg2.connect(
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                )
                cursor = conn.cursor()
                cursor.execute("""
                                    UPDATE user_tokens
                                    SET tokens_remaining = 0
                                    WHERE wp_user_id = %s
                                """, (wp_user_id,))
                conn.commit()
                cursor.close()
                conn.close()

        elif data['status'] == 'active':
            # Pour tous les plans actifs, attribuer les permissions
            app.logger.error(f"Client actif: attribution des permissions pour client_id: {client_id}")
            added, removed = assign_permissions(client_id, data['subscription_level'])
            app.logger.error(f"Permissions updated: {added} added, {removed} removed")

            # !IMPORTANT!  Vérifier si l'utilisateur était précédemment dans un état inactif
            # Ne forcer la réinitialisation des tokens que pour les plans payants
            # OU pour les nouveaux utilisateurs qui n'avaient jamais eu d'abonnement
            is_paid_plan = not is_free_subscription_plan(subscription_level)
            is_new_user = old_subscription is None

            if is_paid_plan or is_new_user:
                # Forcer la réinitialisation complète des tokens
                initialize_user_tokens(wp_user_id, subscription_level, force_reset=True)
                app.logger.error(f"Réactivation du compte: tokens réinitialisés pour l'utilisateur {wp_user_id}")
            else:
                # Pour les plans gratuits qui se réactivent, conserve le nombre de tokens actuel
                # Mais réinitialise les dates de rechargement
                initialize_user_tokens(wp_user_id, subscription_level, force_reset=False)
                app.logger.error(f"Réactivation plan gratuit: tokens conservés pour l'utilisateur {wp_user_id}")

        return jsonify({"message": "Mise à jour de l'abonnement réussie"}), 200

    except ValueError as e:
        app.logger.error(f"Date parsing error: {str(e)}")
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD HH:MM:SS"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/tokens/initialize', methods=['POST'])
@token_required
def initialize_user_tokens_endpoint(wp_user_id):
    try:
        data = request.get_json()

        # Vérifier si wp_user_id correspond au token
        if str(wp_user_id) != str(data.get('wp_user_id')):
            return jsonify({"error": "User ID mismatch"}), 403

        subscription_level = data.get('subscription_level')

        # Initialiser les tokens
        tokens = initialize_user_tokens(wp_user_id, subscription_level)

        if tokens is not None:
            return jsonify({
                "success": True,
                "tokens_allocated": tokens,
                "message": "Tokens initialized successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to initialize tokens"
            }), 500

    except Exception as e:
        app.logger.error(f"Error in token initialization: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred"
        }), 500

@app.route('/api/user/tokens', methods=['GET'])
@token_required
def get_user_tokens_endpoint(wp_user_id):
    try:
        app.logger.error(f"Requête /api/user/tokens reçue pour l'utilisateur {wp_user_id}")
        # Récupérer les informations sur les tokens
        tokens_data = get_user_tokens_details(wp_user_id)

        if tokens_data:
            return jsonify({
                "success": True,
                "data": tokens_data
            })
        else:
            return jsonify({
                "success": False,
                "error": "Impossible de récupérer les informations sur les tokens"
            }), 500

    except Exception as e:
        app.logger.error(f"Error getting token information: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Une erreur est survenue lors de la récupération des informations sur les tokens"
        }), 500


def get_user_tokens_details(wp_user_id):
    """
    Récupère des informations détaillées sur les tokens d'un utilisateur
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        cursor = conn.cursor()

        # Récupérer les informations sur les tokens
        cursor.execute("""
            SELECT tokens_remaining, next_refill, last_refill
            FROM user_tokens 
            WHERE wp_user_id = %s
        """, (wp_user_id,))

        result = cursor.fetchone()

        if not result:
            # Si l'utilisateur n'a pas d'entrée dans user_tokens, initialiser avec des valeurs par défaut
            tokens_remaining = 150
            next_refill = datetime.now() + timedelta(days=3)
            last_refill = datetime.now()

            # Insérer ces valeurs dans la base de données
            cursor.execute("""
                INSERT INTO user_tokens (wp_user_id, tokens_remaining, next_refill, last_refill)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (wp_user_id) DO NOTHING
            """, (wp_user_id, tokens_remaining, next_refill, last_refill))

            conn.commit()
        else:
            tokens_remaining, next_refill, last_refill = result

        # Récupérer le niveau d'abonnement
        cursor.execute("""
            SELECT subscription_level FROM clients WHERE wp_user_id = %s
        """, (wp_user_id,))

        client_result = cursor.fetchone()
        subscription_level = client_result[0] if client_result else "basic"

        # Déterminer les limites de tokens en fonction du niveau d'abonnement
        token_limits = {
            "basic": 150,
            "premium": 500,
            "pro": 1500
        }

        token_limit = token_limits.get(subscription_level, 150)

        cursor.close()
        conn.close()

        return {
            "tokens_remaining": tokens_remaining,
            "next_refill": next_refill.isoformat() if next_refill else None,
            "last_refill": last_refill.isoformat() if last_refill else None,
            "token_limit": token_limit,
            "subscription_level": subscription_level,
            "token_costs": {
                "dall-e": 20,
                "midjourney": 30,
                "enhance-image": 15,
                "generate-image": 15
            }
        }

    except Exception as e:
        app.logger.error(f"Error getting token details: {str(e)}")
        return None
@app.route('/generate_image', methods=['GET', 'POST'])
@token_required
def generate_image(wp_user_id):
    print("Function called")  # Basic print to verify function entry
    app.logger.error("Function entered")
    app.logger.error("=== Token Debug ===")
    app.logger.error(f"Raw URL: {request.url}")
    app.logger.error(f"Query params: {request.args}")
    app.logger.error(f"Token from URL: {request.args.get('token')}")
    app.logger.error(f"WP User ID: {wp_user_id}")
    app.logger.error("=== Generate Image Function Start ===")
    app.logger.error(f"wp_user_id received: {wp_user_id}, type: {type(wp_user_id)}")
    app.logger.error(f"Request method: {request.method}")
    app.logger.error(f"Request args: {request.args}")
    try:
        app.logger.error("Attempting to get client...")
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.info(f"Client lookup result: {client}")

        # Initial checks
        if not client:
            app.logger.error("Client not found!")
            return jsonify({"error": "Client not found"}), 404

        if not check_client_permission(client["id"], "generate_image"):
            app.logger.debug(f"User {wp_user_id} permission denied for genreate an image .")
            return jsonify({"error": "Permission denied"}), 403

        app.logger.info(f"Method: {request.method}")

        # GET request - show form
        if request.method == 'GET':
            app.logger.error("Returning template...")
            #récupère le nombre de tokens restants
            tokens_required = get_user_tokens(wp_user_id)
            history = image_manager.get_user_history(wp_user_id, "generated")
            return render_template('generate-image.html', history=history, LOGIN_URL=LOGIN_URL)

        # POST request - generate image
        if request.method == 'POST':
            prompt = request.form.get('prompt')
            model = request.form.get('model', 'dall-e')  # Default to DALL-E if not specified

            # Gallery share community option
            share_to_gallery = request.form.get('share_to_gallery', 'false').lower() == 'true'

            app.logger.error(f"Processing request - Model: {model}, Prompt: {prompt}")

            app.logger.error(f"Formulaire complet: {request.form}")

            additional_params = {}
            additional_params_str = request.form.get('additional_params')
            app.logger.error(f"additional_params_str brut: {additional_params_str}")
            if additional_params_str:
                try:
                    additional_params = json.loads(additional_params_str)
                    app.logger.error(f"Received additional params: {additional_params}")
                except json.JSONDecodeError as e:
                    app.logger.error(f"Error parsing additional params:{e}")
                    app.logger.error(f"Contenu problématique: '{additional_params_str}'")

            if model == 'midjourney' and not additional_params:
                app.logger.error("Modèle Midjourney sans paramètres. Utilisation d'un dictionnaire vide.")
                additional_params = {}

            if not prompt:
                return jsonify({"message": "Create what inspires you!"}), 400

            # Déterminer le coût en tokens selon le modèle
            token_cost = 20 if model == 'dall-e' else 30  # 30 tokens pour midjourney

            # Vérifier si l'utilisateur a suffisamment de tokens
            if not use_tokens(wp_user_id, token_cost):
                return jsonify({
                    "error": "Insufficient tokens",
                    "message": "You don't have enough tokens for this operation.",
                    "tokens_required": token_cost,
                    "tokens_available": get_user_tokens(wp_user_id)
                }), 200

            try:
                # Utiliser le gestionnaire de modèles
                result = ai_manager.generate_image_sync(model, prompt, additional_params)
                app.logger.error(f"Generation result: {result}")

                if result['success']:
                    if model == 'midjourney':
                        # Mise à jour des métadonnées pour inclure wp_user_id
                        metadata_key = f"midjourney_task:{result['task_id']}"
                        redis_client.hset(metadata_key, 'wp_user_id', str(wp_user_id).encode('utf-8'))

                        if share_to_gallery:
                            redis_client.hset(metadata_key, 'share_to_gallery', b'true')

                        # Ajouter également une entrée dans l'index utilisateur
                        user_history_key = f"user:{wp_user_id}:midjourney_history"
                        redis_client.sadd(user_history_key, result['task_id'])
                        redis_client.expire(user_history_key, TEMP_STORAGE_DURATION)

                        redis_client.expire(metadata_key, TEMP_STORAGE_DURATION)
                        # Pour Midjourney, on retourne le task_id pour le suivi
                        return jsonify({
                            'success': True,
                            'status': 'processing',
                            'task_id': result['task_id']
                        })
                    else:
                        # Pour DALL-E, télécharger et stocker l'image
                        image_response = requests.get(result['image_url'])

                        if image_response.status_code == 200:
                            # Préparer les métadonnées
                            metadata = {
                                'type': b'generated',
                                'prompt': prompt.encode('utf-8'),
                                'timestamp': datetime.now().isoformat().encode('utf-8'),
                                'model': model.encode('utf-8'),
                                'parameters': json.dumps({
                                    'model': model,
                                    'size': '1024x1024'
                                }).encode('utf-8')
                            }

                            # Stocker l'image avec ses métadonnées
                            image_key = image_manager.store_temp_image(
                                wp_user_id,
                                image_response.content,
                                metadata
                            )

                            # Vérification du stockage
                            stored_image = image_manager.get_image(image_key)
                            app.logger.error(f"Image storage verification: {'Success' if stored_image else 'Failed'}")

                            stored_metadata = image_manager.redis.hgetall(f"{image_key}:meta")
                            app.logger.error(f"Stored metadata verification: {stored_metadata}")

                            # Share in the gallery if required
                            if share_to_gallery:
                                # Add image to the gallery
                                decoded_metadata = {
                                    'prompt': prompt,
                                    'timestamp': datetime.now().isoformat(),
                                    'model': model,
                                    'parameters': {
                                        'model': model,
                                        'size': '1024x1024'
                                    }
                                }
                                gallery_item_id = gallery_manager.add_to_gallery(image_key, decoded_metadata, wp_user_id)
                                app.logger.error(f"Image ajoutée à la galerie avec l'ID: {gallery_item_id}")

                            return jsonify({
                                'success': True,
                                'image_key': image_key,
                                'image_url': f"/image/{image_key}",
                                'share_to_gallery': share_to_gallery,
                            })
                        else:
                            refund_tokens(wp_user_id, token_cost)
                            return jsonify({'error': 'Failed to download image'}), 500
                else:
                    refund_tokens(wp_user_id, token_cost)
                    return jsonify({'error': result.get('error', 'Failed to generate image')}), 500

            except openai.OpenAIError as e:
                #en cas d'erreur rembourser le client
                refund_tokens(wp_user_id, token_cost)
                error_message = str(e)
                if "billing_hard_limit_reached" in error_message:
                    error_message = "Billing limit reached. Please check your OpenAI account."
                return jsonify({'error': error_message}), 500
            except requests.RequestException as e:
                refund_tokens(wp_user_id, token_cost)
                app.logger.error(f"Request error: {str(e)}")
                return jsonify({'error': 'Failed to connect to image service'}), 500

        return jsonify({"error": "Invalid request method"}), 400

    except Exception as e:
        app.logger.error(f"Unexpected error in generate_image: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


#@app.route('/download-image/<image_key>')
#@token_required
#def download_image(wp_user_id, image_key):
    #try:
        #app.logger.info(f"Tentative de téléchargement de l'image: {image_key}")

        # Récupérer l'image
        #image_data = storage_manager.get(image_key, 'images')

        #if not image_data:
            #app.logger.error(f"Image non trouvée pour téléchargement: {image_key}")
            #return "Image non trouvée", 404

        # Récupérer les métadonnées pour un nom de fichier personnalisé
        #metadata = storage_manager.get_metadata(image_key)
        #prompt = ''

        #if metadata:
            #prompt_value = metadata.get('prompt', '')
            #if prompt_value:
                #prompt = prompt_value[:30].replace(' ', '_')

        #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Déterminer l'extension basée sur le type de contenu
        #extension = 'png'  # Par défaut
        #try:
            #import magic
            #mime = magic.Magic(mime=True)
            #content_type = mime.from_buffer(image_data)
           # if content_type == 'image/jpeg':
               # extension = 'jpg'
            #elif content_type == 'image/gif':
                #extension = 'gif'
        #except ImportError:
           # pass  # Utiliser l'extension par défaut

        #return send_file(
           # BytesIO(image_data),
            #mimetype=f'image/{extension}',
            #as_attachment=True,
            #download_name=f"aitheway_image_{prompt}_{timestamp}.{extension}"
        #)

    #except Exception as e:
        #app.logger.error(f"Erreur lors du téléchargement de l'image: {str(e)}")
        #return "Erreur lors du téléchargement", 500

@app.route('/download-image', methods=['POST'])
def download_image():
    image_url = request.form['image_url']
    img_format = request.form['format'].upper()
    is_valid, img_format = validate_image_format(img_format)
    if not is_valid:
        return jsonify({'error': img_format}), 400

    # Chemin de l'image locale, supposant que l'image est déjà téléchargée et stockée
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_url.split('/')[-1])
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img_io = BytesIO()
        img.save(img_io, img_format)
        img_io.seek(0)
        return send_file(img_io, mimetype=f'image/{img_format.lower()}', download_name=f'generated_image.{img_format.lower()}')

    return jsonify({'error': 'Image not found'}), 404

@app.route('/upload-enhance', methods=['GET', 'POST'])
@token_required
def upload_enhance(wp_user_id):

    app.logger.error("Function entered")
    app.logger.error("=== Token Debug ===")
    app.logger.error(f"Raw URL: {request.url}")
    app.logger.error(f"Query params: {request.args}")
    app.logger.error(f"Token from URL: {request.args.get('token')}")
    app.logger.error(f"WP User ID: {wp_user_id}")
    app.logger.error("=== Upload Enhance Function Start ===")
    app.logger.error(f"wp_user_id received: {wp_user_id}, type: {type(wp_user_id)}")
    app.logger.error(f"Request method: {request.method}")
    app.logger.error(f"Request args: {request.args}")
    app.logger.error(f"Headers: {dict(request.headers)}")
    client = get_client_by_wp_user_id(wp_user_id)
    # vérifie si le client existe dans la base de données
    #test template error
    token = request.args.get('token')
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    app.logger.error(f"Is AJAX request: {is_ajax}")

    if not token:
        app.logger.error("Token not found!")
        if is_ajax:
            app.logger.error("Returning JSON error response")
            return jsonify({
                "error": "authentification error",
                "message": "Please log in to continue",
                "redirect_url": LOGIN_URL
            }), 401
        else:
            app.logger.error("Returning page with error flag")
            return render_template(
                'upload-enhance.html',
                show_auth_error=True,
                LOGIN_URL=LOGIN_URL,
            )
    try:
        app.logger.error("Attempting to get client...")
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.info(f"Client lookup result: {client}")

        if not client:
            app.logger.error("client non trouvé !")
            return jsonify({"error": "Client non trouvé"}), 404
        # Vérifie si le client existe et a la permission d'utiliser cette route
        if not check_client_permission(client["id"], "upload_enhance"):
            app.logger.error("Permission refusée pour l'amélioration d'image")
            return jsonify({"error": "Permission refusée pour l'upload et l'amélioration d'image"}), 403

        if request.method == 'OPTIONS':
            app.logger.error("OPTIONS request received")
            return '', 204

        if request.method == 'GET':
            app.logger.error("GET request - Returning template...")
            app.logger.error("GET request - reach history user...")
            history = image_manager.get_user_history(wp_user_id, "enhanced")
            return render_template(
                'upload-enhance.html',
                history=history
            )

        if request.method == 'POST':
            app.logger.error("POST request received")
            app.logger.error(f"Files: {request.files}")
            app.logger.error(f"Form: {request.form}")
            if 'file' not in request.files:
                app.logger.error("No file part in the request")
                return jsonify({'error': 'No file part in the request'}), 400

            file = request.files['file']
            app.logger.error(f'File received: %s', file.filename)

            if file and file.content_length > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({
                    'error': 'File too large',
                    'message': 'Le fichier ne doit pas dépasser 20 MB',
                }), 413

            if file.filename == '':
                app.logger.error("No selected file")
                return jsonify({'error': 'No selected file'}), 400

            if file and allowed_file(file.filename):

                # Stocke l'image originale
                image_data = file.read()

                # Stockage de l'image originale
                original_metadata = {
                    'type': 'original',
                    'filename': secure_filename(file.filename),
                    'status': 'pending',  # pour suivre l'état de l'amélioration
                    'timestamp': datetime.now().isoformat()
                }
                original_key = image_manager.store_temp_image(
                    wp_user_id,
                    image_data,
                    original_metadata
                )
                # Create BytesIO object from the file
                file_object = BytesIO(image_data)
                enhanced_url = enhance_image_quality(file_object)

                if enhanced_url:
                    # Télécharger l'image améliorée
                    enhanced_response = requests.get(enhanced_url)
                    if enhanced_response.status_code == 200:
                        # Stockage de l'image améliorée
                        enhanced_metadata = {
                            'type': 'enhanced',
                            'original_key': original_key,
                            'timestamp': datetime.now().isoformat()
                        }
                        enhanced_key = image_manager.store_temp_image(
                            wp_user_id,
                            enhanced_response.content,
                            enhanced_metadata
                        )

                        return jsonify({
                            'success': True,
                            'original_key': original_key,
                            'enhanced_key': enhanced_key,
                            'original_url': f"/image/{original_key}",
                            'enhanced_url': f"/image/{enhanced_key}",
                            'status': 'enhanced'
                        })
                return jsonify({'error': 'Failed to enhance image'}), 500

            return jsonify({'error': 'Invalid file type'}), 400

        return jsonify({"error": "Method not allowed"}), 405

    except Exception as e:
        app.logger.error(f"Unexpected error in /upload_enhance: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def enhance_image_quality(file_path):
    try:
        if not deep_ai_api_key:
            app.logger.error("DeepAI API key is not set")
            return None

        # If file_path is BytesIO, read it directly
        if isinstance(file_path, BytesIO):
            image_file = file_path
        else:
            # If it's a path, open the file
            image_file = open(file_path, 'rb')

        try:
            app.logger.error("Attempting to contact DeepAI API...")
            response = requests.post(
                'https://api.deepai.org/api/waifu2x',
                files={'image': image_file},
                headers={'api-key': deep_ai_api_key},
                timeout=60
            )
            app.logger.error(f"DeepAI API Status Code: {response.status_code}")
            app.logger.error(f"DeepAI API Response Headers: {response.headers}")

            if response.status_code != 200:
                app.logger.error(f"DeepAI API Error: Status {response.status_code}")
                app.logger.error(f"Response content: {response.text}")
                return None

            result = response.json()
            app.logger.debug(f"Response from DeepAI: {result}")
            return result.get('output_url')

        except requests.exceptions.Timeout:
            app.logger.error("Timeout lors de la connexion à DeepAI")
            return None
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Erreur de requête DeepAI: {str(e)}")
            return None
        except Exception as e:
            app.logger.error(f"Erreur inattendue: {str(e)}")
            return None

        finally:
            # Close the file if it was opened from a path
            if not isinstance(file_path, BytesIO) and image_file:
                image_file.close()

    except Exception as e:
        app.logger.error(f"Unexpected error contacting DeepAI: {str(e)}")
        return None

@app.route('/generate_audio', methods=['GET', 'POST'])
@token_required
def generate_audio(wp_user_id):
    app.logger.error("=== Generate Audio Function Start ===")
    app.logger.error(f"wp_user_id received: {wp_user_id}, type: {type(wp_user_id)}")
    app.logger.error(f"Request method: {request.method}")

    try:
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.info(f"Client lookup result: {client}")

        # Initial checks
        if not client:
            app.logger.error("Client not found!")
            return jsonify({"error": "Client not found"}), 404

        if not check_client_permission(client["id"], "generate_audio"):
            app.logger.debug(f"User {wp_user_id} permission denied for generating audio.")
            return jsonify({"error": "Permission denied"}), 403

        # GET request - show form
        if request.method == 'GET':
            tokens_remaining = get_user_tokens(wp_user_id)
            history = audio_manager.get_user_history(wp_user_id)
            return render_template('generate-audio.html', history=history, tokens_remaining=tokens_remaining)

        # POST request - generate audio
        if request.method == 'POST':
            text = request.form.get('text')
            voice = request.form.get('voice', 'alloy')  # Valeur par défaut
            speed = request.form.get('speed', '1.0')

            if not text:
                return jsonify({"message": "Please enter text to convert to speech"}), 400

            # vériifer la longueur du texte
            if len(text) > 4096:
                return jsonify({
                    "error": "Text too long",
                    "message": "The text exceeds the maximum length of 4096 characters"
                }), 400

            # Déterminer le coût en tokens - adapté selon votre tarification
            token_cost = 15  # Exemple, à ajuster selon votre modèle économique

            # Vérifier si l'utilisateur a suffisamment de tokens
            if not use_tokens(wp_user_id, token_cost):
                return jsonify({
                    "error": "Insufficient tokens",
                    "message": "You don't have enough tokens for this operation.",
                    "tokens_required": token_cost,
                    "tokens_available": get_user_tokens(wp_user_id)
                }), 200

            try:
                # paramètres supplémentaires
                additional_params = {
                    "voice": voice,
                    "speed": float(speed)
                }
                # Utiliser le gestionnaire de modèles
                result = ai_manager.generate_image_sync("tts", text, additional_params)

                if result['success']:

                    # Stocker l'audio dans Redis
                    metadata = {
                        'type': b'audio',
                        'text': text.encode('utf-8'),
                        'timestamp': datetime.now().isoformat().encode('utf-8'),
                        'voice': voice.encode('utf-8'),
                        'speed': speed.encode('utf-8'),
                        'model': b'tts'
                    }

                    # On pourrait créer un AudioManager similaire à ImageManager
                    # Mais pour simplifier, utilisons le même ImageManager.  Peut chnager à l'avenir bien sur
                    audio_key = audio_manager.store_temp_audio(
                        wp_user_id,
                        result['audio_data'],
                        metadata
                    )

                    # obtenir le nombre de token restant après génération
                    tokens_remaining = get_user_tokens(wp_user_id)

                    return jsonify({
                        'success': True,
                        'audio_key': audio_key,
                        'audio_url': f"/audio/{audio_key}",
                        'tokens_remaining': tokens_remaining
                    })
                else:
                    refund_tokens(wp_user_id, token_cost)
                    return jsonify({'error': result.get('error', 'Failed to generate audio')}), 500

            except Exception as e:
                refund_tokens(wp_user_id, token_cost)
                app.logger.error(f"Error in audio generation: {str(e)}")
                return jsonify({'error': str(e)}), 500

        return jsonify({"error": "Invalid request method"}), 400

    except Exception as e:
        app.logger.error(f"Unexpected error in generate_audio: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/audio/history', methods=['GET'])
@token_required
def get_audio_history(wp_user_id):
    try:
        app.logger.error(f"Fetching audio history for user {wp_user_id}")

        # Récupérer le numéro de page et le nombre d'éléments par page depuis l'URL
        page = request.args.get('page', 1, type=int)
        per_page = min(50, request.args.get('per_page', 20, type=int))

        # Récupérer l'historique complet
        history = audio_manager.get_user_history(wp_user_id)

        # Calculer les indices pour la pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Extraire la portion demandée de l'historique
        items = history[start_idx:end_idx] if history else []
        total_items = len(history) if history else 0

        # Retourner la réponse formatée
        return {
            'success': True,
            'data': {
                'items': items,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': total_items,
                    'has_more': total_items > end_idx
                }
            }
        }, 200

    except Exception as e:
        app.logger.error(f"Error fetching audio history: {str(e)}")
        return {
            'success': False,
            'error': 'Failed to fetch audio history'
        }, 500


@app.route('/download-audio/<audio_key>')
@token_required
def download_audio(wp_user_id, audio_key):
    try:
        app.logger.info(f"Tentative de téléchargement de l'audio: {audio_key} par l'utilisateur {wp_user_id}")

        # Récupérer l'audio
        audio_data = storage_manager.get(audio_key, 'audio')

        if not audio_data:
            app.logger.error(f"Audio non trouvé pour téléchargement: {audio_key}")
            return "Audio non trouvé", 404

        # Récupérer les métadonnées pour un nom de fichier personnalisé
        metadata = storage_manager.get_metadata(audio_key)
        text = ''
        voice = 'default'

        if metadata:
            if 'text' in metadata:
                text = metadata['text'][:30].replace(' ', '_')
            if 'voice' in metadata:
                voice = metadata['voice']

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        return send_file(
            BytesIO(audio_data),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=f"aitheway_audio_{voice}_{text}_{timestamp}.mp3"
        )

    except Exception as e:
        app.logger.error(f"Erreur lors du téléchargement de l'audio: {str(e)}")
        return "Erreur lors du téléchargement", 500

# Initialize gallery handler
gallery_manager = None

def init_gallery_manager():
    global gallery_manager
    from gallery_manager import GalleryManager
    gallery_manager = GalleryManager(redis_client)


@app.route('/gallery', methods=['GET'])
@token_required
def get_gallery(wp_user_id):
    """
    Récupère et affiche la galerie publique
    """
    try:
        app.logger.error(f"Debut get_gallery pour : {wp_user_id}")
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier les permissions (tous les utilisateurs peuvent voir le contenu)
        if not check_client_permission(client["id"], "view_content"):
            return jsonify({"error": "Permission denied"}), 403

        # Récupérer les paramètres de requête pour le filtrage et la pagination
        page = request.args.get('page', 1, type=int)
        per_page = min(50, request.args.get('per_page', 20, type=int))

        # Paramètres de filtrage
        environment = request.args.get('environment', 'tous')
        movement = request.args.get('movement', 'tous')
        duration = request.args.get('duration', 'tous')
        model = request.args.get('model', 'tous')

        # Paramètre de tri
        sort_by = request.args.get('sort', 'recent')

        # Construire les filtres
        filters = {
            'environment': environment,
            'movement': movement,
            'duration': duration,
            'model': model
        }

        app.logger.info(f"Filtres: {filters}")
        app.logger.info(f"Type de filters: {type(filters)}")
        app.logger.info(f"Type de gallery_manager: {type(gallery_manager)}")
        app.logger.info(f"Type de gallery_manager.redis: {type(gallery_manager.redis)}")

        # Récupérer les éléments de la galerie
        gallery_data = gallery_manager.get_gallery_items(
            page=page,
            per_page=per_page,
            filters=filters,
            sort_by=sort_by
        )

        # Pour une requête AJAX, retourner JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'data': gallery_data
            })

        # Pour une requête normale, rendre le template HTML
        return render_template('gallery.html',
                               gallery_data=gallery_data,
                               current_filters=filters,
                               current_sort=sort_by
                               )

    except Exception as e:
        app.logger.error(f"Error fetching gallery: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch gallery'
        }), 500


@app.route('/gallery/item/<gallery_item_id>', methods=['GET'])
@token_required
def get_gallery_item(wp_user_id):
    """
    Version simplifiée de la galerie pour diagnostiquer le problème
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier les permissions
        if not check_client_permission(client["id"], "view_content"):
            return jsonify({"error": "Permission denied"}), 403

        # Créer des données de galerie statiques
        mock_gallery_data = {
            'items': [
                {
                    'gallery_id': 'mock-item-1',
                    'prompt': 'Exemple de prompt pour la galerie',
                    'image_url': '/static/assets/placeholder-image.jpg',
                    'model': 'dall-e',
                    'shared_by': wp_user_id,
                    'timestamp': datetime.now().isoformat(),
                    'likes': 5,
                    'views': 20,
                    'size': 'large',
                    'featured': 'True'
                },
                {
                    'gallery_id': 'mock-item-2',
                    'prompt': 'Un autre exemple pour démontrer la galerie',
                    'image_url': '/static/assets/placeholder-image2.jpg',
                    'model': 'midjourney',
                    'shared_by': wp_user_id,
                    'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
                    'likes': 10,
                    'views': 35,
                    'size': 'medium',
                    'featured': 'False'
                }
            ],
            'pagination': {
                'current_page': 1,
                'per_page': 20,
                'total_items': 2,
                'has_more': False
            }
        }

        # Pour une requête AJAX, retourner JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'data': mock_gallery_data
            })

        # Paramètres de filtrage pour le template
        current_filters = {
            'environment': 'tous',
            'movement': 'tous',
            'duration': 'tous',
            'model': 'tous'
        }

        # Pour une requête normale, rendre le template HTML
        return render_template('gallery.html',
                               gallery_data=mock_gallery_data,
                               current_filters=current_filters,
                               current_sort='recent',
                               current_date=datetime.now().strftime('%Y-%m-%d')
                               )

    except Exception as e:
        app.logger.error(f"Error in simplified gallery: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error in gallery: {str(e)}'
        }), 500


@app.route('/gallery/like/<gallery_item_id>', methods=['POST'])
@token_required
def like_gallery_item(wp_user_id, gallery_item_id):
    """
    Ajoute ou retire un like sur un élément de la galerie
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier les permissions
        if not check_client_permission(client["id"], "view_content"):
            return jsonify({"error": "Permission denied"}), 403

        # Ajouter/retirer le like
        result = gallery_manager.like_gallery_item(gallery_item_id, wp_user_id)

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error liking gallery item: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to like gallery item'
        }), 500


@app.route('/gallery/featured', methods=['GET'])
@token_required
def get_featured_gallery_items(wp_user_id):
    """
    Récupère les éléments mis en avant de la galerie
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier les permissions
        if not check_client_permission(client["id"], "view_content"):
            return jsonify({"error": "Permission denied"}), 403

        # Récupérer le nombre d'éléments à afficher
        limit = request.args.get('limit', 6, type=int)

        # Récupérer les éléments mis en avant
        featured_items = gallery_manager.get_featured_items(limit=limit)

        return jsonify({
            'success': True,
            'data': featured_items
        })

    except Exception as e:
        app.logger.error(f"Error fetching featured gallery items: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch featured gallery items'
        }), 500


# Routes administratives (sécurisées)
@app.route('/admin/gallery/feature/<gallery_item_id>', methods=['POST'])
@token_required
def feature_gallery_item(wp_user_id, gallery_item_id):
    """
    Marque un élément comme mis en avant (admin seulement)
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier si l'utilisateur est un administrateur (à adapter selon votre système)
        is_admin = False  # Par défaut, aucun utilisateur n'est admin

        # Logique pour vérifier si l'utilisateur est un administrateur
        # Par exemple, vous pourriez avoir une table ou un champ dans votre base de données
        # is_admin = check_if_admin(wp_user_id)

        if not is_admin:
            return jsonify({"error": "Admin access required"}), 403

        # Récupérer l'état de mise en avant depuis la requête
        featured = request.json.get('featured', True)

        # Mettre à jour l'élément
        success = gallery_manager.feature_gallery_item(gallery_item_id, featured)

        if not success:
            return jsonify({"error": "Item not found or update failed"}), 404

        return jsonify({
            'success': True,
            'message': f"Item {'featured' if featured else 'unfeatured'} successfully"
        })

    except Exception as e:
        app.logger.error(f"Error featuring gallery item: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to update item'
        }), 500


@app.route('/admin/gallery/delete/<gallery_item_id>', methods=['DELETE'])
@token_required
def delete_gallery_item(wp_user_id, gallery_item_id):
    """
    Supprime un élément de la galerie (admin ou propriétaire seulement)
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        if not client:
            return jsonify({"error": "Client not found"}), 404

        # Vérifier si l'utilisateur est un administrateur
        is_admin = False  # Par défaut, aucun utilisateur n'est admin
        # is_admin = check_if_admin(wp_user_id)

        # Supprimer l'élément (la fonction vérifiera si l'utilisateur est autorisé)
        success = gallery_manager.delete_from_gallery(gallery_item_id, wp_user_id, is_admin)

        if not success:
            return jsonify({"error": "Item not found or permission denied"}), 404

        return jsonify({
            'success': True,
            'message': "Item deleted successfully"
        })

    except Exception as e:
        app.logger.error(f"Error deleting gallery item: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to delete item'
        }), 500


# Routes de service mises à jour
@app.route('/image/<image_key>')
def serve_image(image_key):
    app.logger.info(f"Tentative d'accès à l'image: {image_key}")

    # Récupérer l'image
    image_data = storage_manager.get(image_key, 'images')

    if not image_data:
        app.logger.error(f"Image non trouvée: {image_key}")
        return "Image non trouvée", 404

    app.logger.info(f"Image trouvée: {image_key}, taille: {len(image_data)} octets")

    # Déterminer le type MIME de l'image
    content_type = 'image/png'  # Type par défaut

    # Tenter de déterminer le type à partir des premiers octets
    if image_data[:2] == b'\xff\xd8':
        content_type = 'image/jpeg'
    elif image_data[:8] == b'\x89PNG\r\n\x1a\n':
        content_type = 'image/png'
    elif image_data[:6] in (b'GIF87a', b'GIF89a'):
        content_type = 'image/gif'

    # Servir l'image avec les bons en-têtes
    response = send_file(
        BytesIO(image_data),
        mimetype=content_type,
        conditional=True,
        etag=True
    )

    # En-têtes pour optimisation du cache
    response.headers['Cache-Control'] = 'public, max-age=86400'  # 24h

    return response

@app.route('/audio/<audio_key>')
def serve_audio(audio_key):
    app.logger.info(f"Tentative d'accès à l'audio: {audio_key}")

    # Récupérer les métadonnées
    metadata = storage_manager.get_metadata(audio_key)

    # Tentative de récupération du fichier audio
    audio_data = storage_manager.get(audio_key, 'audio')

    if not audio_data:
        app.logger.error(f"Audio non trouvé: {audio_key}")
        return "Audio non trouvé", 404

    app.logger.info(f"Audio trouvé: {audio_key}, taille: {len(audio_data)} octets")

    # Servir l'audio avec les bons en-têtes
    response = send_file(
        BytesIO(audio_data),
        mimetype='audio/mpeg',
        conditional=True,
        etag=True,
        as_attachment=False
    )

    # En-têtes essentiels pour le streaming
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'public, max-age=3600'

    return response

# --- Video generation route ---


@app.route('/generate_video', methods=['GET', 'POST'])
@token_required
def generate_video(wp_user_id):
    """
    GET: Affiche la page de génération de vidéo
    POST: Génère une nouvelle vidéo
    """
    app.logger.error(f"=== Generate Video Request ===")
    app.logger.error(f"Form data: {request.form}")
    app.logger.error(f"Files: {request.files}")

    client = get_client_by_wp_user_id(wp_user_id)
    app.logger.error(f"Client: {client}")

    # Version GET: Affiche le formulaire
    if request.method == 'GET':
        tokens_remaining = get_user_tokens(wp_user_id)
        history = video_manager.get_user_video_history(wp_user_id)
        return render_template('generate-video.html', history=history, tokens_remaining=tokens_remaining, client=client)

    # Version POST: Traite la demande de génération
    elif request.method == 'POST':
        # Récupérer les données du formulaire
        prompt = request.form.get('prompt')
        model = request.form.get('model', 'T2V-01-Director')

        # Si une image a été sélectionnée, changer le modèle pour Image-to-Video
        if 'first_frame_image' in request.files and request.files['first_frame_image'].filename:
            app.logger.error(f"Image detected, using for reference")
            model = 'I2V-01-Director'

        # Vérifier permissions et tokens
        if not client or not check_client_permission(client["id"], "generate_video"):
            return jsonify({"error": "Permission denied"}), 403

        # Coût: 100 tokens
        token_cost = 100
        if not use_tokens(wp_user_id, token_cost):
            return jsonify({
                "error": "Insufficient tokens",
                "tokens_required": token_cost,
                "tokens_available": get_user_tokens(wp_user_id)
            }), 200

        # Préparer les paramètres additionnels
        additional_params = {
            "prompt_optimizer": request.form.get('prompt_optimizer', 'false').lower() == 'true',
        }

        # Dans la section de traitement d'image de votre route /generate_video:
        if 'first_frame_image' in request.files and request.files['first_frame_image'].filename:
            file = request.files['first_frame_image']
            app.logger.error(f"Processing image file: {file.filename}")

            if file and allowed_file(file.filename):
                try:
                    # Lire l'image
                    image_data = file.read()
                    app.logger.error(f"Image data read, size: {len(image_data)} bytes")

                    # Traiter l'image avec la fonction améliorée
                    from minimax_video import process_image_for_minimax
                    result = process_image_for_minimax(image_data, max_size_mb=5)

                    if not result["success"]:
                        # Rembourser les tokens à l'utilisateur en cas d'erreur
                        refund_tokens(wp_user_id, token_cost)

                        app.logger.error(f"Image processing error: {result['error_code']} - {result['error_message']}")

                        # Renvoyer une réponse d'erreur structurée au frontend
                        return jsonify({
                            'error': result['error_message'],
                            'error_code': result['error_code'],
                            'success': False,
                            'tokens_refunded': token_cost
                        }), 400

                    # Image traitée avec succès
                    encoded_image = result["image"]

                    # Ajouter aux paramètres
                    additional_params["first_frame_image"] = encoded_image
                    additional_params["subject_reference"] = True

                    # Informations sur les corrections appliquées
                    if result.get("corrections") and result["corrections"] != ["Aucune correction nécessaire"]:
                        app.logger.error(f"Image corrections applied: {result['corrections']}")

                    app.logger.error(
                        f"Image encoded successfully: format={result['format']}, dimensions={result['dimensions']}, "
                        f"ratio={result['ratio']}, size={result['size_mb']}MB, encoded={result['encoded_size_mb']}MB")

                except Exception as e:
                    # Rembourser les tokens en cas d'erreur
                    refund_tokens(wp_user_id, token_cost)

                    app.logger.error(f"Error processing reference image: {str(e)}")
                    app.logger.error(traceback.format_exc())

                    return jsonify({
                        'error': f"Erreur lors du traitement de l'image: {str(e)}",
                        'error_code': 'PROCESSING_ERROR',
                        'success': False,
                        'tokens_refunded': token_cost
                    }), 500
            else:
                # Rembourser les tokens si le format du fichier n'est pas autorisé
                refund_tokens(wp_user_id, token_cost)

                allowed_extensions = ", ".join(app.config['ALLOWED_EXTENSIONS'])
                return jsonify({
                    'error': f"Format de fichier non autorisé. Formats acceptés: {allowed_extensions}",
                    'error_code': 'INVALID_FILE_FORMAT',
                    'success': False,
                    'tokens_refunded': token_cost
                }), 400

        # Options de mouvement de caméra
        camera_movement = request.form.get('camera_movement')
        if camera_movement and camera_movement != 'none':
            prompt = f"{prompt} [{camera_movement}]"

        # Log des paramètres finaux (sans l'image pour éviter de saturer les logs)
        safe_params = additional_params.copy()
        if "first_frame_image" in safe_params:
            safe_params["first_frame_image"] = f"[BASE64_ENCODED_IMAGE - {len(safe_params['first_frame_image'])} chars]"

        app.logger.error(f"Final parameters: model={model}, prompt='{prompt}', additional_params={safe_params}")

        # Envoyer la demande via l'AI Manager
        result = ai_manager.generate_image_sync("minimax-video", prompt, additional_params)

        # Traiter la réponse
        if result.get('success'):
            task_id = result.get('task_id')

            if not task_id:
                app.logger.error(f"No task_id in Minimax response")
                return jsonify({'error': 'No task_id in MiniMax response'}), 500

            app.logger.error(f"Got task_id from MiniMax: {task_id}")

            app.logger.error(
                f"About to store video metadata -user_id: {wp_user_id}, task_id: {task_id}, additional_params: {safe_params}")

            video_key = video_manager.store_video_metadata(wp_user_id, task_id, prompt, model, safe_params)

            app.logger.error(f"Stored video metadata with key: {video_key}")

            test_key = video_manager.get_video_by_task_id(task_id)
            app.logger.error(f"Verification - retrieved video key: {test_key}")
            return jsonify({
                'success': True,
                'status': 'processing',
                'task_id': task_id,
                'video_key': video_key,
                'is_premium': client.get('subscription_level') in ['28974']
            })
        else:
            refund_tokens(wp_user_id, token_cost)
            return jsonify({'error': result.get('error')}), 500
@app.route('/check_video_status/<task_id>')
@token_required
def check_video_status(wp_user_id, task_id=None):
    """
    Vérifie l'état d'une tâche de génération de vidéo
    """
    app.logger.error("=== Starting Video Status Check ===")
    app.logger.error(f"User ID: {wp_user_id}")
    app.logger.error(f"Task ID: {task_id}")

    if not task_id:
        return jsonify({
            "success": False,
            "status": "error",
            "error": "No task_id provided"
        }), 400

    try:
        # Récupérer l'état actuel
        video_info = video_manager.get_video_by_task_id(task_id)

        # Si aucune information n'est disponible, créer une entrée minimale
        if not video_info:
            app.logger.error(f"No local information for task {task_id}, creating temporary record")
            video_key = video_manager.store_video_metadata(
                wp_user_id,
                task_id,
                "Untitled video",
                "T2V-01-Director"
            )
            video_info = video_manager.get_video_by_task_id(task_id)
            if not video_info:
                video_info = {'status': 'processing', 'video_key': video_key}

        # Si complète et le fichier est stocké, retourner les informations
        if video_info.get('status') == 'completed' and video_info.get('file_stored'):
            return jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": "completed",
                    "video_key": video_info.get('video_key'),
                    "video_url": f"/video/{video_info.get('video_key')}",
                    "prompt": video_info.get('prompt', '')
                }
            })

        # Sinon, vérifier auprès de l'API MiniMax
        generator = ai_manager.generators.get("minimax-video")
        if not generator:
            return jsonify({
                'success': False,
                'error': "Video generator not available"
            }), 500

        status_response = generator.generator.check_generation_status(task_id)

        # Traiter la réponse
        if status_response.get('success'):
            remote_status = status_response.get('status', '')
            file_id = status_response.get('file_id', '')

            # Si terminée, récupérer l'URL et mettre à jour
            if remote_status == "Success" and file_id:
                download_url = generator.generator.get_download_url(file_id)

                if download_url:
                    video_manager.update_video_status(task_id, 'completed', file_id, download_url)

                    # Tenter de télécharger immédiatement la vidéo
                    video_data = video_manager.download_from_url(download_url)
                    if video_data and video_info.get('video_key'):
                        video_manager.store_video_file(video_info.get('video_key'), video_data)

                    return jsonify({
                        "success": True,
                        "data": {
                            "status": "completed",
                            "video_key": video_info.get('video_key'),
                            "video_url": f"/video/{video_info.get('video_key')}"
                        }
                    })
                else:
                    return jsonify({
                        "success": True,
                        "data": {
                            "status": "url_error",
                            "message": "Video processing complete but download URL not available"
                        }
                    })

            # Sinon, mettre à jour le statut
            status_mapping = {
                "Processing": "processing",
                "Pending": "pending",
                "Running": "processing",
                "Fail": "failed",
                "Error": "failed"
            }

            local_status = status_mapping.get(remote_status, remote_status)
            video_manager.update_video_status(task_id, local_status)

            return jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": local_status,
                    "remote_status": remote_status,
                    "video_key": video_info.get('video_key'),
                    "progress": status_response.get('progress', 0)
                }
            })
        else:
            # Erreur dans la récupération du statut depuis l'API
            return jsonify({
                "success": False,
                "error": status_response.get('error', 'Failed to check task status'),
                "data": {
                    "status": "error",
                    "video_key": video_info.get('video_key')
                }
            })

    except Exception as e:
        app.logger.error(f"Error checking video status: {e}")
        app.logger.error(traceback.format_exc())

        # Retourner une réponse d'erreur avec le plus d'informations possible
        return jsonify({
            "success": False,
            "error": str(e),
            "task_id": task_id,
            "data": {
                "status": "error",
                "message": "An unexpected error occurred while checking video status"
            }
        }), 500

@app.route('/video/<video_key>')
def serve_video(video_key):
    """
    Sert la vidéo pour lecture dans le navigateur
    """
    app.logger.info(f"Tentative d'accès à la vidéo: {video_key}")

    # Récupérer les métadonnées en premier pour déterminer la stratégie
    metadata = storage_manager.get_metadata(video_key)

    # Variables pour le suivi
    video_data = None
    source = None

    # 1. Tentative de récupération de la vidéo
    video_data = storage_manager.get(video_key, 'videos')

    if video_data:
        source = "storage"
        app.logger.error(f"Vidéo trouvée dans le stockage: {video_key}")

    # 2. Si pas trouvé mais que nous avons des métadonnées avec une URL de téléchargement
    elif metadata and 'download_url' in metadata:
        download_url = metadata['download_url']
        app.logger.info(f"Tentative de récupération via URL {download_url}")

        # Utiliser la nouvelle version améliorée de download_from_url qui gère les URLs expirées
        video_data = video_manager.download_from_url(download_url)

        if video_data:
            # Stocker pour les prochaines requêtes
            video_manager.store_video_file(video_key, video_data)
            source = "remote_url"
            app.logger.error(f"Vidéo récupérée et stockée via URL: {video_key}")

    # 3. Si toujours pas trouvée et que nous avons un task_id, essayer via l'API
    elif metadata and 'task_id' in metadata:
        task_id = metadata['task_id']
        app.logger.info(f"Tentative de régénération d'URL via task_id: {task_id}")

        try:
            # Récupérer le générateur MiniMax
            from ai_models import create_ai_manager
            ai_manager = create_ai_manager()
            minimax_generator = ai_manager.generators.get("minimax-video")

            if minimax_generator:
                # Vérifier le statut pour obtenir le file_id
                status_response = minimax_generator.generator.check_generation_status(task_id)

                if status_response.get('success') and status_response.get('file_id'):
                    file_id = status_response.get('file_id')

                    # Obtenir une nouvelle URL de téléchargement
                    new_url = minimax_generator.generator.get_download_url(file_id)

                    if new_url:
                        app.logger.info(f"Nouvelle URL générée: {new_url}")

                        # Mettre à jour les métadonnées
                        metadata['download_url'] = new_url
                        storage_manager.store_metadata(video_key, metadata)

                        # Télécharger la vidéo
                        video_data = video_manager.download_from_url(new_url)

                        if video_data:
                            # Stocker pour les prochaines requêtes
                            video_manager.store_video_file(video_key, video_data)
                            source = "regenerated_url"
                            app.logger.info(f"Vidéo récupérée via nouvelle URL: {video_key}")

        except Exception as e:
            app.logger.error(f"Erreur lors de la régénération d'URL: {str(e)}")

    if not video_data:
        # Si la vidéo n'est pas trouvée mais les métadonnées existent
        metadata = storage_manager.get_metadata(video_key)
        if metadata and 'download_url' in metadata:
            # Tentative explicite de téléchargement
            download_url = metadata['download_url']
            video_data = video_manager.download_from_url(download_url)

            if video_data:
                # Stocker immédiatement
                video_manager.store_video_file(video_key, video_data)
                app.logger.info(f"Video {video_key} téléchargée et stockée à la demande")
            else:
                # Si l'URL est expirée, chercher par task_id
                task_id = metadata.get('task_id')
                if task_id:
                    # Essayer de régénérer l'URL via l'API
                    from ai_models import create_ai_manager
                    ai_manager = create_ai_manager()
                    generator = ai_manager.generators.get("minimax-video")
                    if generator:
                        status = generator.generator.check_generation_status(task_id)
                        if status.get('file_id'):
                            new_url = generator.generator.get_download_url(status.get('file_id'))
                            if new_url:
                                video_data = video_manager.download_from_url(new_url)
                                if video_data:
                                    video_manager.store_video_file(video_key, video_data)

    app.logger.info(f"Vidéo trouvée ({source}): {video_key}, taille: {len(video_data)} octets")

    # Servir la vidéo avec les bons en-têtes pour streaming
    response = send_file(
        BytesIO(video_data),
        mimetype='video/mp4',
        conditional=True,  # Support des requêtes Range
        etag=True,  # Support du cache via ETag
        as_attachment=False
    )

    # En-têtes essentiels pour le streaming
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'public, max-age=3600'

    return response
@app.route('/download-video/<video_key>')
@token_required
def download_video(wp_user_id, video_key):
    """Télécharge la vidéo (même logique que serve_video mais en tant que téléchargement)"""
    try:
        app.logger.info(f"Téléchargement de vidéo demandé: {video_key}")

        # Récupérer les métadonnées d'abord
        metadata = storage_manager.get_metadata(video_key)
        video_data = None

        # Tentative de récupération de la vidéo
        video_data = storage_manager.get(video_key, 'videos')

        # Si la vidéo n'est pas trouvée mais que nous avons une URL de téléchargement
        if not video_data and metadata and 'download_url' in metadata:
            app.logger.info(f"Tentative de téléchargement depuis: {metadata['download_url']}")
            video_data = video_manager.download_from_url(metadata['download_url'])

            if video_data:
                # Stocker pour usage futur
                video_manager.store_video_file(video_key, video_data)
                app.logger.info(f"Vidéo retéléchargée et stockée: {video_key}")

        if not video_data:
            app.logger.error(f"Vidéo non trouvée pour téléchargement: {video_key}")
            return "Vidéo non trouvée", 404

        # Préparer un nom de fichier approprié
        prompt = metadata.get('prompt', '').replace(' ', '_')[:30] if metadata else 'video'
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        app.logger.info(f"Envoi de la vidéo pour téléchargement: {video_key}")

        return send_file(
            BytesIO(video_data),
            mimetype='video/mp4',
            as_attachment=True,
            download_name=f"aitheway_video_{prompt}_{timestamp}.mp4"
        )

    except Exception as e:
        app.logger.error(f"Erreur lors du téléchargement de la vidéo: {str(e)}")
        return "Erreur lors du téléchargement", 500
@app.route('/api/video/history', methods=['GET'])
@token_required
def get_video_history(wp_user_id):
    """
    Retourne l'historique des vidéos générées par l'utilisateur
    """
    page = request.args.get('page', 1, type=int)
    per_page = min(50, request.args.get('per_page', 10, type=int))

    history = video_manager.get_user_video_history(wp_user_id, page, per_page)

    return jsonify({
        'success': True,
        'data': {
            'items': history,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_items': len(history),
                'has_more': len(history) >= per_page
            }
        }
    })


@app.route('/video-thumbnail/<video_key>')
def serve_video_thumbnail(video_key):
    """Génère et sert une vignette GIF animée pour une vidéo avec qualité améliorée et transition fluide (palindrome)"""
    try:
        request_id = str(uuid.uuid4())[:8]  # Identifiant unique pour cette requête
        app.logger.error(f"[REQ-{request_id}] === THUMBNAIL DEBUG ===")
        app.logger.error(f"[REQ-{request_id}] Génération de vignette demandée pour: {video_key}")

        try:
            ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True)
            app.logger.error(f"[REQ-{request_id}] FFmpeg disponible: {ffmpeg_version.split()[2]}")
        except Exception as e:
            app.logger.error(f"[REQ-{request_id}] Erreur lors de la vérification de FFmpeg: {str(e)}")

        # Forcer la régénération pour déboguer (à retirer en production)
        force_regenerate = request.args.get('force', '0') == '1'
        app.logger.error(f"[REQ-{request_id}] Force regenerate: {force_regenerate}")

        # Vérifier si une vignette GIF existe déjà - ATTENTION : UTILISEZ LA BONNE CLÉ !
        thumbnail_key = f"{video_key}:thumbnail:gif:v4"  # Version 4 pour forcer une nouvelle génération
        app.logger.error(f"[REQ-{request_id}] Recherche vignette avec clé: {thumbnail_key}")
        thumbnail_data = None if force_regenerate else storage_manager.get(thumbnail_key, 'images')
        app.logger.error(f"[REQ-{request_id}] Vignette existante trouvée: {thumbnail_data is not None}")

        if thumbnail_data and not force_regenerate:
            app.logger.error(f"Utilisation de la vignette existante pour {video_key}")
            return send_file(
                BytesIO(thumbnail_data),
                mimetype='image/gif'
            )

        # Récupérer la vidéo
        app.logger.error(f"Récupération de la vidéo depuis le stockage")
        video_data = storage_manager.get(video_key, 'videos')

        if video_data:
            app.logger.error(f"Vidéo récupérée, taille: {len(video_data)} octets")
        else:
            app.logger.error(f"Vidéo non trouvée dans le stockage: {video_key}")

        if not video_data:
            # Si vidéo non trouvée, servir une image par défaut
            app.logger.error(f"Video not found for thumbnail: {video_key}")
            default_path = os.path.join(app.static_folder, 'assets', 'img', 'video-placeholder.png')
            if os.path.exists(default_path):
                app.logger.error(f"Utilisation de l'image placeholder par défaut")
                with open(default_path, 'rb') as f:
                    return send_file(
                        BytesIO(f.read()),
                        mimetype='image/png'
                    )
            return "Video not found", 404

        if len(video_data) > MAX_VIDEO_SIZE_FOR_THUMBNAIL:
            app.logger.error(f"Video too large for thumbnail: {len(video_data)} bytes")
            return redirect(url_for('static', filename='assets/img/large-video.png'))

        # Créer des dossiers temporaires si nécessaires
        temp_dir = "/tmp/video_thumbnails"
        os.makedirs(temp_dir, exist_ok=True)
        app.logger.error(f"Dossier temporaire créé: {temp_dir}")

        # Générer des noms de fichiers uniques
        unique_id = str(uuid.uuid4())
        temp_video_path = f"{temp_dir}/{unique_id}.mp4"
        temp_gif_path = f"{temp_dir}/{unique_id}.gif"
        advanced_gif_path = f"{temp_dir}/{unique_id}_advanced.gif"
        app.logger.error(
            f"Fichiers temporaires: video={temp_video_path}, gif={temp_gif_path}, advanced={advanced_gif_path}")

        # Écrire temporairement la vidéo
        with open(temp_video_path, 'wb') as f:
            f.write(video_data)
        app.logger.error(f"Vidéo écrite sur disque: {temp_video_path}")

        # Paramètres améliorés pour le GIF
        width = 640  # Augmenté pour meilleure qualité
        fps = 10  # Légèrement réduit pour meilleur encodage

        # Obtenir la durée de la vidéo
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            temp_video_path
        ]

        app.logger.error(f"Executing FFprobe command: {' '.join(probe_cmd)}")
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        app.logger.error(f"FFprobe result stdout: {probe_result.stdout}")
        app.logger.error(f"FFprobe result stderr: {probe_result.stderr}")
        app.logger.error(f"FFprobe return code: {probe_result.returncode}")

        if probe_result.returncode != 0:
            app.logger.error(f"FFprobe error: {probe_result.stderr}")
            raise Exception(f"FFprobe failed: {probe_result.stderr}")

        duration_info = json.loads(probe_result.stdout)
        video_duration = float(duration_info['format']['duration'])
        app.logger.error(f"Durée vidéo détectée: {video_duration} secondes")

        # Calculer les paramètres pour le GIF
        gif_duration = min(4.0, video_duration / 2)  # Maximum 4 secondes ou moitié de la vidéo

        # Commande FFmpeg améliorée pour la qualité
        advanced_gif_cmd = [
            'ffmpeg',
            '-i', temp_video_path,
            '-t', str(gif_duration),
            '-filter_complex',
            f'[0:v]fps={fps},scale={width}:-1:flags=lanczos:sws_dither=none,split[forward][temp];'
            f'[temp]reverse[backward];'
            f'[forward][backward]concat=n=2:v=1:a=0,split[s0][s1];'
            f'[s0]palettegen=stats_mode=diff:max_colors=256[p];'
            f'[s1][p]paletteuse=dither=floyd_steinberg:bayer_scale=5:diff_mode=rectangle',
            '-loop', '0',
            '-y',
            advanced_gif_path
        ]

        app.logger.error(f"Executing advanced FFmpeg command: {' '.join(advanced_gif_cmd)}")
        advanced_result = subprocess.run(advanced_gif_cmd, capture_output=True, text=True)
        app.logger.error(f"Advanced FFmpeg return code: {advanced_result.returncode}")

        if advanced_result.stderr:
            app.logger.error(f"Advanced FFmpeg stderr: {advanced_result.stderr}")

        # Si la commande avancée échoue, essayer la commande de base comme fallback
        if advanced_result.returncode != 0:
            app.logger.error(f"Advanced FFmpeg command failed, trying basic command as fallback")
            basic_gif_cmd = [
                'ffmpeg',
                '-i', temp_video_path,
                '-t', str(gif_duration),
                '-filter_complex',
                # Configuration simplifiée qui préserve mieux les couleurs
                f'fps={fps},scale={width}:-1:flags=bilinear,split[forward][temp];'
                f'[temp]reverse[backward];'
                f'[forward][backward]concat=n=2:v=1:a=0,split[s0][s1];'
                f'[s0]palettegen=max_colors=256:stats_mode=diff[p];'  # Mode diff au lieu de single
                f'[s1][p]paletteuse=dither=none',  # Pas de dithering pour éviter le grain
                '-loop', '0',
                '-y',
                temp_gif_path
            ]

            app.logger.error(f"Executing basic FFmpeg command: {' '.join(basic_gif_cmd)}")
            basic_result = subprocess.run(basic_gif_cmd, capture_output=True, text=True)
            app.logger.error(f"Basic FFmpeg return code: {basic_result.returncode}")

            if basic_result.returncode != 0:
                app.logger.error(f"Basic FFmpeg error: {basic_result.stderr}")
                raise Exception(f"All FFmpeg commands failed")

            final_gif_path = temp_gif_path
            is_palindrome = False
        else:
            final_gif_path = advanced_gif_path
            is_palindrome = True
            app.logger.error(f"Advanced GIF (palindrome) généré avec succès: {advanced_gif_path}")

        # Lire le GIF généré
        app.logger.error(f"Lecture du GIF final: {final_gif_path}")
        with open(final_gif_path, 'rb') as f:
            thumbnail_data = f.read()
        app.logger.error(f"GIF chargé en mémoire, taille: {len(thumbnail_data)} octets")

        # Stocker pour les futures demandes
        metadata = {
            'type': 'thumbnail',
            'format': 'gif',
            'video_key': video_key,
            'created_at': datetime.now().isoformat(),
            'width': width,
            'fps': fps,
            'duration': gif_duration * 2 if is_palindrome else gif_duration,
            'is_palindrome': is_palindrome
        }
        app.logger.error(f"Stockage du GIF avec métadonnées: {metadata}")
        storage_manager.store(thumbnail_key, thumbnail_data, metadata, 'images')
        app.logger.error(f"Thumbnail générée et stockée avec clé: {thumbnail_key}")

        # Nettoyer
        app.logger.error("Nettoyage des fichiers temporaires")
        for path in [temp_video_path, temp_gif_path, advanced_gif_path]:
            if os.path.exists(path):
                os.remove(path)
                app.logger.error(f"Fichier supprimé: {path}")

        # Servir
        app.logger.error(f"Envoi du GIF au client, taille: {len(thumbnail_data)} octets")
        return send_file(
            BytesIO(thumbnail_data),
            mimetype='image/gif'
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"ERREUR CRITIQUE lors de la génération du GIF: {str(e)}")
        app.logger.error(f"Traceback complet: {error_trace}")
        # En cas d'erreur, rediriger vers une image par défaut
        return redirect(url_for('static', filename='assets/svg/video.svg'))
@app.route('/save-image', methods=['POST'])
@token_required
def save_image(wp_user_id):
    image_key = request.form.get('image_key')
    if not image_key:
        app.logger.error("Image key not found")
        return jsonify({'error': 'No image key provided'}), 400

    try:
        image_data = image_manager.get_image(image_key)
        if not image_data:
            app.logger.error(f"Image not found: {image_key}")
            return jsonify({'error': 'Image not found'}), 404

        # Sauvegarde de manière permanente
        permanent_key = image_manager.save_image(wp_user_id, image_data)

        return jsonify({
            'success': True,
            'permanent_key': image_key,
            'image_url': f"/image/{permanent_key}"
        })
    except Exception as e:
        app.logger.error(f"Error saving image: {str(e)}")
        return jsonify({'error': 'Failed to save image'}), 500

@app.route('/chat/send', methods=['POST'])
@token_required
def send_message(wp_user_id):
    """
    Envoie un message avec potentiellement une image
    """
    try:
        message_data = {
            'text': request.form.get('text', ''),
            'type': request.form.get('type', 'text')
        }

        # Gestion d'une image si présente
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                # Stockage de l'image
                image_content = file.read()
                image_key = image_manager.store_temp_image(
                    wp_user_id,
                    image_content,
                    "chat_image"
                )
                message_data['image_key'] = image_key
                message_data['type'] = 'image'

       # Stockage du message
        message_id = chat_manager.store_message(wp_user_id, message_data)

        return jsonify({
            'success': True,
            'message_id': message_id,
            'message': chat_manager.get_message(message_id)
        })
    except Exception as e:
        app.logger.error(f"Error sending message: {str(e)}")
        return jsonify({'error': 'Failed to send message'}), 500


@app.route('/api/chat/history/generated', methods=['GET'])
@token_required
def get_chat_history(wp_user_id):
   response, status_code = get_paginated_history(wp_user_id, "generated")
   return jsonify(response), status_code

@app.route('/api/chat/history/enhanced', methods=['GET'])
@token_required
def get_enhanced_history(wp_user_id):
    # Utilise la fonction utilitaire pour les images améliorées
    response, status_code = get_paginated_history(wp_user_id, "enhanced")
    # Transformer les données pour le format attendu par le frontend
    if response['success'] and 'data' in response:
        for item in response['data']['items']:
            if item.get('model') == 'midjourney':
                # S'assurer que les URLs sont complètes
                for image in item.get('images', []):
                    if 'url' in image and not image['url'].startswith('http'):
                        image['url'] = f"/image/{image['key']}"

    return jsonify(response), status_code

@app.route('/check_midjourney_status/<task_id>')
@token_required
def check_midjourney_status(wp_user_id, task_id=None):
    app.logger.error("=== Starting Midjourney Status Check ===")
    app.logger.error(f"User ID: {wp_user_id}")
    app.logger.error(f"Task ID: {task_id}")

    app.logger.error("Checking task_id parameter")

    # Verify task_id
    if not task_id:
        app.logger.error("No task_id provided, returning 400")
        return jsonify({
            "success": False,
            "status": "error",
            "error": "No task_id provided"
        }), 400

    try:
        app.logger.error(f"Preparing to fetch task data for {task_id}")
        metadata_key = f"midjourney_task:{task_id}"
        group_key = f"midjourney_group:{task_id}"

        app.logger.error(f"Getting data with keys: {metadata_key}, {group_key}")

        # Get task data
        task_data = redis_client.hgetall(metadata_key)
        group_data_bytes = redis_client.get(group_key)

        app.logger.error(f"Task data from Redis: {task_data}")
        app.logger.error(f"Group data from Redis: {group_data_bytes}")

        if not task_data:
            return jsonify({
                "success": False,
                "error": "Task not found"
            }), 404

        # Decode task data
        status = task_data.get(b'status', b'processing').decode('utf-8')
        prompt = task_data.get(b'prompt', b'').decode('utf-8')

        # Prepare base response
        response = {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": status,
                "prompt": prompt,
                "timestamp": task_data.get(b'timestamp', b'').decode('utf-8')
            }
        }

        # Add group data if available
        if group_data_bytes:
            try:
                group_info = json.loads(group_data_bytes)

                # Process image URLs to ensure they have the correct format
                processed_images = []

                for img in group_info.get('images', []):
                    # Make a copy to avoid modifying the original
                    image_copy = img.copy()

                    # Ensure URL is accessible or transform it if needed
                    if 'url' in image_copy and not image_copy['url'].startswith('http'):
                        image_copy['url'] = f"https://{image_copy['url']}"

                    # Add any missing fields needed by the frontend
                    if 'key' not in image_copy:
                        image_copy['key'] = f"midjourney_image:{task_id}:{img.get('variation_number', 0)}"

                    # Add width and height if not present (frontend might expect these)
                    if 'width' not in image_copy:
                        image_copy['width'] = 1024
                    if 'height' not in image_copy:
                        image_copy['height'] = 1024

                    processed_images.append(image_copy)

                # Replace images array with processed one
                group_info['images'] = processed_images

                response["data"].update({
                    "initial_grid": group_info.get('initial_grid'),
                    "images": group_info.get('images', []),
                    "group_status": group_info.get('status', 'pending')
                })

            except json.JSONDecodeError:
                app.logger.error(f"Error decoding group data JSON for task {task_id}")
                # Continue without group data

        # If status is error, include error message
        if status == 'error':
            response["data"]["error"] = task_data.get(b'error', b'Unknown error').decode('utf-8')

        app.logger.error(f"Sending response: {response}")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error checking status for task {task_id}: {str(e)}")
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/midjourney_callback', methods=['POST'])
def midjourney_callback():
    app.logger.error("=== Midjourney Callback Started ===")
    try:
        data = request.get_json()
        app.logger.error(f"Received callback data: {data}")

        # Validation of received data
        required_fields = ['task_id', 'image_url', 'variation_number', 'type']
        if not all(field in data for field in required_fields):
            app.logger.error("Missing required fields in callback data")
            return jsonify({"error": "Missing required fields"}), 400

        task_id = data['task_id']
        image_url = data['image_url']
        variation_number = data['variation_number']
        callback_type = data['type']  # 'initial' or 'upscale'

        # Get Redis keys
        metadata_key = f"midjourney_task:{task_id}"
        group_key = f"midjourney_group:{task_id}"

        # Verify the task exists
        if not redis_client.exists(metadata_key):
            app.logger.error(f"Task {task_id} not found")
            return jsonify({"error": "Task not found"}), 404

        # Use a Redis pipeline for atomic operations
        pipe = redis_client.pipeline()

        if callback_type == 'initial':
            # Store initial grid
            prompt = redis_client.hget(metadata_key, b'prompt').decode('utf-8')
            group_data = {
                'task_id': task_id,
                'prompt': prompt,
                'initial_grid': image_url,
                'images': [],
                'timestamp': datetime.now().isoformat(),
                'status': 'pending'
            }
            pipe.setex(
                group_key,
                TEMP_STORAGE_DURATION,
                json.dumps(group_data)
            )
            pipe.execute()
            status = "processing"

        elif callback_type == 'upscale':
            # Get existing group data
            group_data_bytes = redis_client.get(group_key)

            if not group_data_bytes:
                app.logger.error(f"Group data missing for task {task_id}")
                return jsonify({"error": "Group data missing"}), 404

            group_info = json.loads(group_data_bytes)

            # Add new upscaled image with all necessary metadata
            new_image = {
                'url': image_url,
                'variation_number': variation_number,
                'choice': variation_number,  # For frontend compatibility
                'timestamp': datetime.now().isoformat(),
                'type': 'upscale',
                'key': f"midjourney_image:{task_id}:{variation_number}"
            }

            # Check if this variation already exists, replace it if it does
            existing_index = None
            for i, img in enumerate(group_info['images']):
                if img.get('variation_number') == variation_number:
                    existing_index = i
                    break

            if existing_index is not None:
                group_info['images'][existing_index] = new_image
            else:
                group_info['images'].append(new_image)

            # Update status if all images are received (4 upscales)
            if len(group_info['images']) >= 4:
                group_info['status'] = 'completed'
                pipe.hset(metadata_key, 'status', b'completed')
                status = "completed"

                # Nouveau: Vérifier si l'image doit être partagée dans la galerie
                share_to_gallery = redis_client.hget(metadata_key, b'share_to_gallery')
                wp_user_id = redis_client.hget(metadata_key, b'wp_user_id')

                if share_to_gallery and wp_user_id:
                    share_to_gallery = share_to_gallery.decode('utf-8') if isinstance(share_to_gallery,
                                                                                      bytes) else share_to_gallery
                    wp_user_id = wp_user_id.decode('utf-8') if isinstance(wp_user_id, bytes) else wp_user_id

                    if share_to_gallery == 'true':
                        # Partager chaque image dans la galerie
                        prompt = group_info.get('prompt', '')

                        for img in group_info['images']:
                            # Préparer les métadonnées pour la galerie
                            metadata = {
                                'prompt': prompt,
                                'timestamp': datetime.now().isoformat(),
                                'model': 'midjourney',
                                'parameters': {
                                    'model': 'midjourney',
                                    'task_id': task_id,
                                    'variation_number': img.get('variation_number', 0)
                                }
                            }

                            # Clé de l'image
                            image_key = img.get('key', f"midjourney_image:{task_id}:{img.get('variation_number', 0)}")

                            # Ajouter à la galerie
                            gallery_item_id = gallery_manager.add_to_gallery(image_key, metadata, wp_user_id)
                            app.logger.error(
                                f"Midjourney image {image_key} ajoutée à la galerie avec l'ID: {gallery_item_id}")

                app.logger.error(f"All 4 images received for task {task_id}, marking as completed")
            else:
                status = "processing"

            # Save group modifications
            pipe.setex(
                group_key,
                TEMP_STORAGE_DURATION,
                json.dumps(group_info)
            )

            # Also store the image data individually
            image_key = f"midjourney_image:{task_id}:{variation_number}"
            pipe.setex(
                image_key,
                TEMP_STORAGE_DURATION,
                json.dumps({
                    'url': image_url,
                    'task_id': task_id,
                    'variation_number': variation_number
                })
            )

            pipe.execute()

        return jsonify({
            "success": True,
            "task_id": task_id,
            "type": callback_type,
            "status": status
        })

    except Exception as e:
        app.logger.error(f"Error in callback: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/midjourney/templates/<template_type>', methods=['GET'])
@token_required
def get_midjourney_template(wp_user_id, template_type):
    subject = request.args.get('subject', '')

    # Récupérer les options spécifiques pour ce modèle
    template_options = {}
    for param in request.args:
        if param not in ['subject', 'token']:
            template_options[param] = request.args.get(param)

    # Générer le prompt à partir du modèle
    prompt = MidjourneyTemplateSystem.generate_from_template(
        template_type,
        subject=subject,
        **template_options
    )

    return jsonify({
        'success': True,
        'prompt': prompt,
        'template_type': template_type
    })


@app.route('/api/midjourney/analyze-prompt', methods=['POST'])
@token_required
def analyze_midjourney_prompt(wp_user_id):
    data = request.get_json()
    prompt = data.get('prompt', '')

    # Analyser le prompt
    params = MidjourneyAnalyzer.extract_parameters(prompt)
    clean_prompt = MidjourneyAnalyzer.clean_prompt(prompt)

    return jsonify({
        'success': True,
        'parameters': params,
        'clean_prompt': clean_prompt
    })


@app.route('/api/midjourney/collections', methods=['GET', 'POST'])
@token_required
def manage_collections(wp_user_id):
    if request.method == 'GET':
        # Récupérer les collections de l'utilisateur
        collections = MidjourneyCollection.get_user_collections(redis_client, wp_user_id)
        return jsonify({
            'success': True,
            'collections': collections
        })

    elif request.method == 'POST':
        # Créer une nouvelle collection
        data = request.get_json()
        name = data.get('name', f"Collection {datetime.now().strftime('%Y-%m-%d')}")
        description = data.get('description', '')

        collection_id = MidjourneyCollection.create_collection(
            redis_client,
            wp_user_id,
            name,
            description
        )

        return jsonify({
            'success': True,
            'collection_id': collection_id,
            'message': 'Collection created successfully'
        })


@app.route('/api/midjourney/collections/<collection_id>', methods=['GET', 'POST'])
@token_required
def collection_operations(wp_user_id, collection_id):
    if request.method == 'GET':
        # Récupérer une collection spécifique
        collection = MidjourneyCollection.get_collection(redis_client, collection_id, wp_user_id)
        if not collection:
            return jsonify({
                'success': False,
                'error': 'Collection not found'
            }), 404

        return jsonify({
            'success': True,
            'collection': collection
        })

    elif request.method == 'POST':
        # Ajouter une image à la collection
        data = request.get_json()
        image_key = data.get('image_key')

        if not image_key:
            return jsonify({
                'success': False,
                'error': 'Image key is required'
            }), 400

        success = MidjourneyCollection.add_image_to_collection(
            redis_client,
            collection_id,
            image_key,
            wp_user_id
        )

        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to add image to collection'
            }), 400

        return jsonify({
            'success': True,
            'message': 'Image added to collection successfully'
        })


@app.route('/download-enhanced-image', methods=['POST'])
def download_enhanced_image():
    image_url = request.form['image_url']
    img_format = request.form['format'].upper()
    is_valid, img_format = validate_image_format(img_format)
    if not is_valid:
        return jsonify({'error': img_format}), 400

    # Télécharger l'image améliorée depuis l'URL
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        img = Image.open(BytesIO(image_response.content))
        img_io = BytesIO()
        img.save(img_io, img_format)
        img_io.seek(0)
        return send_file(img_io, mimetype=f'image/{img_format.lower()}', download_name=f'enhanced_image.{img_format.lower()}')

    return jsonify({'error': 'Enhanced image not found'}), 404

# Library asset

@app.route('/library', methods=['GET'])
@token_required
def library_view(wp_user_id):
    """
    Unified library view that shows all user's generated content
    (images, videos, audio) in one place
    """
    try:
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.info(f"Client lookup result: {client}")

        # Initial checks
        if not client:
            app.logger.error("Client not found!")
            return jsonify({"error": "Client not found"}), 404

        if not check_client_permission(client["id"], "view_content"):
            app.logger.debug(f"User {wp_user_id} permission denied for viewing library.")
            return jsonify({"error": "Permission denied"}), 403

        # Get tokens information
        tokens_remaining = get_user_tokens(wp_user_id)

        # Render the library template
        return render_template(
            'library.html',
            tokens_remaining=tokens_remaining
        )
    except Exception as e:
        app.logger.error(f"Unexpected error in library view: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/library', methods=['GET'])
@token_required
def get_library_items(wp_user_id):
    """
    API endpoint to fetch all library items for a user with filtering and pagination
    """
    try:
        # Get request parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(50, request.args.get('per_page', 20, type=int))
        content_type = request.args.get('type', 'all')
        date_filter = request.args.get('date', 'recent')
        search_query = request.args.get('search', '')

        # Calculate pagination indices
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Fetch all content types
        all_items = []

        # 1. Fetch images if requested
        if content_type in ['all', 'image']:
            try:
                images = image_manager.get_user_history(wp_user_id, "generated")

                for img in images:
                    # Handle regular images
                    if 'model' in img and img['model'] != 'midjourney':
                        item = {
                            'id': img.get('key', ''),
                            'type': 'image',
                            'url': f"/image/{img.get('key', '')}",
                            'thumbnail_url': f"/image/{img.get('key', '')}",
                            'description': img.get('prompt', ''),
                            'timestamp': img.get('timestamp', ''),
                            'model': img.get('model', ''),
                            'dimensions': img.get('parameters', {}).get('size', '1024x1024').split('x') if img.get(
                                'parameters') else None
                        }

                        # Add dimensions as an object if available
                        if item['dimensions'] and len(item['dimensions']) == 2:
                            try:
                                item['dimensions'] = {
                                    'width': int(item['dimensions'][0]),
                                    'height': int(item['dimensions'][1])
                                }
                            except (ValueError, IndexError):
                                item['dimensions'] = None

                        all_items.append(item)

                    # Handle Midjourney image groups
                    elif 'model' in img and img['model'] == 'midjourney' and 'images' in img:
                        for midjourney_img in img.get('images', []):
                            item = {
                                'id': midjourney_img.get('key', ''),
                                'type': 'image',
                                'url': f"/image/{midjourney_img.get('key', '')}",
                                'thumbnail_url': f"/image/{midjourney_img.get('key', '')}",
                                'description': img.get('prompt', ''),
                                'timestamp': img.get('timestamp', ''),
                                'model': 'midjourney',
                                'task_id': img.get('task_id', '')
                            }
                            all_items.append(item)
            except Exception as e:
                app.logger.error(f"Error fetching images: {str(e)}")

        # 2. Fetch videos if requested
        if content_type in ['all', 'video']:
            try:
                videos = video_manager.get_user_video_history(wp_user_id)

                for video in videos:
                    if video.get('status') == 'completed':
                        item = {
                            'id': video.get('video_key', ''),
                            'type': 'video',
                            'url': f"/video/{video.get('video_key', '')}",
                            'thumbnail_url': f"/video-thumbnail/{video.get('video_key', '')}",  # Anim thumbnail
                            'description': video.get('prompt', ''),
                            'timestamp': video.get('timestamp', ''),
                            'model': video.get('model', 'default'),
                            'duration': video.get('duration', 0)
                        }
                        all_items.append(item)
            except Exception as e:
                app.logger.error(f"Error fetching videos: {str(e)}")

        # 3. Fetch audio if requested
        if content_type in ['all', 'audio']:
            try:
                audio_files = audio_manager.get_user_history(wp_user_id)

                for audio in audio_files:
                    audio_key = audio.get('audio_key', '') or audio.get('key', '')
                    app.logger.error(f"Audio trouvé: {audio}")
                    if audio_key:
                        item = {
                            'id': audio_key,
                            'type': 'audio',
                            'url': f"/audio/{audio_key}",
                            'thumbnail_url': None,  # Audio doesn't have thumbnails
                            'description': truncate_text(audio.get('text', ''), 100),
                            'timestamp': audio.get('timestamp', ''),
                            'model': 'tts',
                            'voice': audio.get('voice', 'default'),
                            'duration': audio.get('duration', 0),
                            'text': audio.get('text', '')
                        }
                        all_items.append(item)
                    else:
                        app.logger.error(f"Audio ignoré car clé manquante")
            except Exception as e:
                app.logger.error(f"Error fetching audio: {str(e)}")

        # Apply search filter if provided
        if search_query:
            search_query = search_query.lower()
            all_items = [
                item for item in all_items
                if search_query in (item.get('description') or '').lower() or
                   search_query in (item.get('model') or '').lower() or
                   search_query in (item.get('type') or '').lower()
            ]

        # Apply date filter
        if date_filter:
            all_items = sort_by_date(all_items, date_filter)

        # Get total count before pagination
        total_items = len(all_items)

        # Apply pagination
        paginated_items = all_items[start_idx:end_idx] if all_items else []

        # Return response
        return jsonify({
            'success': True,
            'data': {
                'items': paginated_items,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': total_items,
                    'total_pages': (total_items + per_page - 1) // per_page
                }
            }
        })

    except Exception as e:
        app.logger.error(f"Error in library API: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch library items'
        }), 500


# Helper function to sort items by date
def sort_by_date(items, date_filter):
    """Sort items based on date filter"""
    from datetime import datetime, timedelta

    # First, ensure all timestamps are datetime objects
    for item in items:
        if isinstance(item.get('timestamp'), str):
            try:
                item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # If conversion fails, try a common format
                try:
                    item['timestamp'] = datetime.strptime(item['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                except (ValueError, AttributeError):
                    # If all fails, set to epoch start
                    item['timestamp'] = datetime(1970, 1, 1)

    # Apply filtering based on date
    now = datetime.now()

    if date_filter == 'today':
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        items = [item for item in items if item.get('timestamp') >= today_start]
    elif date_filter == 'week':
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        items = [item for item in items if item.get('timestamp') >= week_start]
    elif date_filter == 'month':
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        items = [item for item in items if item.get('timestamp') >= month_start]

    # Sort by timestamp
    if date_filter == 'oldest':
        return sorted(items, key=lambda x: x.get('timestamp', datetime(1970, 1, 1)))
    else:  # recent is default
        return sorted(items, key=lambda x: x.get('timestamp', datetime(1970, 1, 1)), reverse=True)


# Helper function to truncate text
def truncate_text(text, max_length):
    """Truncate text to a specified length"""
    if not text:
        return ''
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'


# Add routes for downloading content

@app.route('/download-image/<image_key>')
@token_required
def download_specific_image(wp_user_id, image_key):
    """Download a specific image by key"""
    try:
        # Fetch the image
        image_data = image_manager.get_image(image_key)
        if not image_data:
            return "Image not found", 404

        # Get metadata for a better filename
        metadata = image_manager.get_image_metadata(image_key)
        prompt = ''
        if metadata:
            prompt_bytes = metadata.get(b'prompt', b'')
            if prompt_bytes:
                prompt = prompt_bytes.decode('utf-8', errors='ignore')[:30].replace(' ', '_')

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"aitheway_image_{prompt}_{timestamp}.png"

        return send_file(
            BytesIO(image_data),
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Error downloading image: {str(e)}")
        return "Error processing download", 500


@app.route('/download-video/<video_key>')
@token_required
def download_specific_video(wp_user_id, video_key):
    """Download a specific video by key"""
    try:
        # Fetch the video
        video_data = video_manager.get_video(video_key)
        if not video_data:
            # Try to fetch from URL if not stored locally
            metadata = storage_manager.get_metadata(video_key)
            if not metadata or not metadata.get('download_url'):
                return "Video not found", 404

            download_url = metadata.get('download_url')
            video_data = video_manager.download_from_url(download_url)

            if not video_data:
                return "Failed to download video", 500

        # Get metadata for a better filename
        metadata = storage_manager.get_metadata(video_key)
        prompt = ''
        if metadata:
            prompt = metadata.get('prompt', '')[:30].replace(' ', '_')

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"aitheway_video_{prompt}_{timestamp}.mp4"

        return send_file(
            BytesIO(video_data),
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Error downloading video: {str(e)}")
        return "Error processing download", 500


@app.route('/download-audio/<audio_key>')
def download_specific_audio(wp_user_id, audio_key):
    """Download a specific audio file by key"""
    try:
        # Fetch the audio
        audio_data = audio_manager.get_audio(audio_key)
        if not audio_data:
            return "Audio not found", 404

        # Get metadata for a custom filename
        metadata = storage_manager.get_metadata(audio_key)
        text = ''
        voice = 'default'

        if metadata:
            if 'text' in metadata:
                text = metadata['text'][:30].replace(' ', '_')
            if 'voice' in metadata:
                voice = metadata['voice']

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = f"aitheway_audio_{voice}_{text}_{timestamp}.mp3"

        return send_file(
            BytesIO(audio_data),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Error downloading audio: {str(e)}")
        return "Error processing download", 500

# ---- TEMP ----


@app.before_request
def log_all_requests():
    app.logger.error("=== Global Request Interceptor ===")
    app.logger.error(f"Endpoint: {request.endpoint}")
    app.logger.error(f"URL Rule: {request.url_rule}")
    app.logger.error(f"View Args: {request.view_args}")
    app.logger.error(f"Method: {request.method}")
    app.logger.error(f"Path: {request.path}")
    app.logger.error(f"Full URL: {request.url}")
    if request.method == 'OPTIONS':
        app.logger.error("OPTIONS request detected!")

@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    app.logger.error(f"=== Catch-all route hit ===")
    app.logger.error(f"Path: {path}")
    app.logger.error(f"Method: {request.method}")
    return jsonify({"error": "Route not found"}), 404


@app.route('/debug_task/<task_id>', methods=['GET'])
def debug_task(task_id):
    metadata_key = f"midjourney_task:{task_id}"
    all_data = redis_client.hgetall(metadata_key)
    return jsonify({
        "task_data": {k.decode('utf-8'): v.decode('utf-8')
                     for k, v in all_data.items()},
        "exists": redis_client.exists(metadata_key),
        "ttl": redis_client.ttl(metadata_key)
    })


#---- END TEMP!----

#---- ADMIN Storage stats lookup ----
@app.route('/admin/storage/stats', methods=['GET'])
@token_required
def storage_stats(wp_user_id):
    """Obtenir des statistiques de stockage pour les administrateurs."""
    # Vérifier si l'utilisateur est admin
    client = get_client_by_wp_user_id(wp_user_id)
    if not client or client.get("id") != 1:  # Adapter selon votre logique d'admin
        return jsonify({"error": "Admin access required"}), 403

    # Obtenir les stats Redis
    redis_keys = len(redis_client.keys('*'))
    redis_info = redis_client.info('memory')
    redis_memory = redis_info.get('used_memory_human', 'Unknown')

    # Obtenir les stats du stockage fichier
    total_files = 0
    total_size = 0

    for content_type in ['images', 'audio', 'video']:
        path = Path(file_storage.base_path) / content_type
        if path.exists():
            # Compter les fichiers récursivement
            files = list(path.glob('**/*'))
            files = [f for f in files if f.is_file()]
            total_files += len(files)

            # Calculer la taille totale
            size = sum(f.stat().st_size for f in files)
            total_size += size

    return jsonify({
        'redis': {
            'keys': redis_keys,
            'memory_usage': redis_memory
        },
        'file_storage': {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }
    })

def init_background_tasks(_app):
    # Import here to avoid circular imports
    from tasks import check_pending_midjourney_tasks

    # Set the app in the tasks module
    import tasks
    tasks.app = _app

    # Start the background task
    loop = asyncio.get_event_loop()
    loop.create_task(check_pending_midjourney_tasks())
    loop.create_task(migrate_old_redis_content(storage_manager))
    return loop

init_gallery_manager()

if __name__ == '__main__':
    background_task = init_background_tasks(app)
    app.run(debug=True, host='127.0.0.1', port=5432)