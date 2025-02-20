import os
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
from urllib.parse import quote
import redis
import json

from ai_models import create_ai_manager
import uuid
from config import redis_client, TEMP_STORAGE_DURATION, PERMANENT_STORAGE_DURATION



# Définir les chemins des dossiers de templates
template_dir = os.path.abspath('templates')
template_temp_dir = os.path.abspath('templates_temp')




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

deep_ai_api_key = os.getenv('DEEPAI_API_KEY')

# Configuration Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # Important pour les données binaires
)

# Durées de conservation
TEMP_STORAGE_DURATION = timedelta(hours=24)  # Pour les uploads temporaires
PERMANENT_STORAGE_DURATION = timedelta(days=30)  # Pour les images sauvegardées



class ImageManager:
    def __init__(self, redis_client):
        self.redis = redis_client
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

            # Pipeline pour les opérations atomiques
            pipe = self.redis.pipeline()

            # Stockage de l'image
            pipe.setex(
                key,
                TEMP_STORAGE_DURATION,
                image_data
            )

            # Stockage des métadonnées si présentes
            if metadata:
                metadata_key = f"{key}:meta"
                app.logger.error(f"Original metadata: {metadata}")

                pipe.hmset(metadata_key, metadata)
                pipe.expire(metadata_key, TEMP_STORAGE_DURATION)

                # Ajouter à l'index utilisateur
                user_index_key = f"user:{user_id}:images"
                pipe.lpush(user_index_key, key)

            results = pipe.execute()
            app.logger.error(f"Pipeline execution results: {results}")

            return key

        except Exception as e:
            app.logger.error(f"Redis storage error: {str(e)}")
            app.logger.error(f"Failed to store image with metadata: {metadata}")
            raise

    def get_user_history(self, user_id, history_type="enhanced"):
        """Récupère l'historique des images d'un utilisateur"""
        app.logger.error(f"Fetching history for user {user_id} with type {history_type}")

        patterns = [
            f"img:temp:image:{user_id}:*",
            "img:temp:image:midjourney:*"
        ]
        images = []
        processed_groups = set()

        all_keys = []
        for pattern in patterns:
            keys = self.redis.keys(pattern)
            all_keys.extend(keys)

        app.logger.error(f"Found all keys: {all_keys}")

        # Filtrer pour ne garder que les clés d'images (pas les métadonnées)
        app.logger.error(f"Found all keys: {all_keys}")
        image_keys = [k for k in all_keys if not k.endswith(b':meta')]
        app.logger.error(f"Filtered image keys: {image_keys}")

        for key in image_keys:
            try:
                # Construire la clé des métadonnées
                metadata_key = f"{key.decode('utf-8')}:meta"
                metadata = self.redis.hgetall(metadata_key)
                app.logger.error(f"Checking metadata for key {metadata_key}: {metadata}")

                # Vérifier si l'image existe toujours
                if not self.redis.exists(key):
                    app.logger.error(f"Image {key} no longer exists, skipping")
                    continue

                # Traitement selon le type d'historique demandé
                if history_type == "generated" and metadata.get(b'type') == b'generated':
                    if metadata.get(b'model', b'').decode('utf-8') == 'midjourney':
                        task_id = metadata.get(b'task_id', b'').decode('utf-8')

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
                            # Décoder les métadonnées essentielles
                            decoded_prompt = metadata.get(b'prompt', b'').decode('utf-8')
                            decoded_timestamp = metadata.get(b'timestamp', b'').decode('utf-8')
                            model = metadata.get(b'model', b'unknown').decode('utf-8')

                            # Vérifier que l'image appartient à l'utilisateur pour les images DALL-E
                            # Pour Midjourney, on accepte toutes les images car elles sont partagées
                            if model != 'midjourney' and f"img:temp:image:{user_id}" not in key.decode('utf-8'):
                                continue

                            # Décoder et parser les paramètres
                            try:
                                parameters_str = metadata.get(b'parameters', b'{}').decode('utf-8')
                                parameters = json.loads(parameters_str)
                            except (json.JSONDecodeError, TypeError):
                                app.logger.error(f"Error parsing parameters for key {key}")
                                parameters = {}

                            # Construire l'objet image
                            image_data = {
                                'key': key.decode('utf-8'),
                                'url': f"/image/{key.decode('utf-8')}",
                                'prompt': decoded_prompt,
                                'timestamp': decoded_timestamp,
                                'parameters': parameters,
                                'model': model
                            }

                            # Ajouter des champs spécifiques à Midjourney si présents
                            if model == 'midjourney':
                                image_data['task_id'] = metadata.get(b'task_id', b'').decode('utf-8')
                                image_data['make_task_id'] = metadata.get(b'make_task_id', b'').decode('utf-8')

                            images.append(image_data)
                            app.logger.error(f"Added generated image to history: {image_data}")

                        except Exception as e:
                            app.logger.error(f"Error processing generated image metadata: {e}")
                            continue

                elif history_type == "enhanced" and metadata.get(b'type') == b'enhanced':
                    try:
                        # Traitement spécifique pour les images améliorées
                        original_key = metadata.get(b'original_key')
                        timestamp = metadata.get(b'timestamp', b'').decode('utf-8')

                        enhanced_data = {
                            'enhanced_key': key.decode('utf-8'),
                            'original_key': original_key.decode('utf-8') if original_key else None,
                            'timestamp': timestamp,
                            'enhanced_url': f"/image/{key.decode('utf-8')}",
                            'original_url': f"/image/{original_key.decode('utf-8')}" if original_key else None
                        }

                        images.append(enhanced_data)
                        app.logger.error(f"Added enhanced image to history: {enhanced_data}")

                    except Exception as e:
                        app.logger.error(f"Error processing enhanced image metadata: {e}")
                        continue

            except Exception as e:
                app.logger.error(f"Error processing image key {key}: {e}")
                continue

            # Trier les images par timestamp (plus récent en premier)
        sorted_images = sorted(images, key=lambda x: x['timestamp'], reverse=True)
        app.logger.error(f"Returning {len(sorted_images)} sorted images")

        return sorted_images

    def save_image(self, user_id, image_data):
        """Sauvegarde une image de manière permanente"""
        key = f"user:{user_id}:images:{datetime.now().timestamp()}"
        self.redis.setex(
            key,
            PERMANENT_STORAGE_DURATION,
            image_data
        )
        return key

    def get_image(self, key):
        """Récupère une image"""
        if not self.redis.exists(key):
            return None
        return self.redis.get(key)

    def delete_image(self, key):
        """Suppression propre avec métadonnées"""
        pipe = self.redis.pipeline()
        pipe.delete(key)
        pipe.delete(f"{key}:meta")
        pipe.execute()

    def get_image_metadata(self, key):
        """Récupère les métadonnées d'une image"""
        metadata_key = f"{key}:metadata"
        return self.redis.hgetall(metadata_key)

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

image_manager = ImageManager(redis_client)


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
        SELECT p.is_allowed, c.expiration_date, c.status
        FROM permissions AS p
        JOIN clients AS c ON p.client_id = c.id
        WHERE p.client_id = %s AND p.action = %s
    """, (client_id, action))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        is_allowed, expiration_date, status = result
        now = datetime.now()
        # Add debug logs for timezone investigation
        app.logger.error(f"""
        Timezone Debug:
        Current server time (now): {now}
        Expiration date from DB: {expiration_date}
        Status: {status}
        Is allowed: {is_allowed}
        """)

        if is_allowed and ((status == 'active' and now <= expiration_date) or (status == 'canceled' and now <= expiration_date)):
            app.logger.error(f"Permission check results - Is Allowed: {is_allowed}, Status: {status}, Valid: {result}")
            return True
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
    # Define subscription mapping with WordPress ID
    subscription_mapping = {
        28974: "basic",  # WordPress Free Test Plan ID
        "FREE Test PLAN": "basic",  # Keep existing mapping for backward compatibility
        "premium_plan": "premium",
        "pro_plan": "pro"
    }

    # Convert subscription_level to int if it's numeric
    if isinstance(subscription_level, str) and subscription_level.isdigit():
        subscription_level = int(subscription_level)

    # Map the subscription level to our internal levels
    mapped_level = subscription_mapping.get(subscription_level, subscription_level)
    app.logger.info(f"Mapping subscription {subscription_level} to {mapped_level}")

    permissions = {
        "basic": ["view_content", "generate_image", "upload_enhance"],
        "premium": ["view_content", "generate_image", "upload_enhance"],
        "pro": ["view_content", "generate_image", "upload_enhance", "access_special_features"]
    }

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("SELECT action FROM permissions WHERE client_id = %s", (client_id,))
    current_permissions = {row[0] for row in cursor.fetchall()}

    required_permissions = set(permissions.get(mapped_level, []))

    for action in required_permissions - current_permissions:
        add_permission(client_id, action)
        app.logger.info(f"Permission '{action}' assigned to client ID {client_id}")

    cursor.close()
    conn.close()
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
        SELECT id FROM clients WHERE status = 'canceled' AND expiration_date <= %s
    """, (datetime.now(),))
    clients_to_revoke = cursor.fetchall()

    for client_id in clients_to_revoke:
        # Appelle la fonction pour révoquer les permissions pour chaque client expiré
        revoke_client_permissions(client_id[0])
        app.logger.info(f"Permissions revoked for client ID {client_id[0]} due to expired subscription.")

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
def index():
    return render_template('index.html')

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
        expiration_date = datetime.strptime(data['expiration_date'], '%Y-%m-%d %H:%M:%S')

        # Check if expiration date has passed
        if expiration_date < datetime.now():
            if data['status'] == 'canceled':
                data['status'] = 'abandoned'
            else:
                data['status'] = 'expired'
            app.logger.info(f"Subscription marked as expired due to past expiration date")

        # Vérifie si le client existe
        client = get_client_by_wp_user_id(wp_user_id)
        app.logger.error(f"Looking up client with wp_user_id: {wp_user_id}")

        if not client:
            # Si le client n'existe pas, ajoute-le avec les permissions de base
            app.logger.info(f"Adding new client with wp_user_id: {wp_user_id}")
            client_id = add_client(
                wp_user_id=wp_user_id,
                email=data['email'],
                subscription_level=data['subscription_level'],
                status=data['status'],
                expiration_date=expiration_date,
            )
        else:
            client_id = client["id"]
            # Met à jour uniquement si le statut d'abonement change
            app.logger.error(f"Updating existing client with id: {client_id} with status: {data['status']}")
            update_client_subscription(
                client_id=client_id,
                subscription_level=data['subscription_level'],
                status=data['status'],
                expiration_date=expiration_date,
            )

            # Attribue les permissions en fonction du niveau d'abonnement
        if data['status'] == 'active':
            app.logger.info(f"Revoking old permissions for client_id: {client_id}")
            revoke_client_permissions(client_id)

            app.logger.error(f"Assigning permissions for client_id: {client_id}")
            assign_permissions(client_id, data['subscription_level'])

        elif data['status'] in ['abandoned', 'expired']:
            app.logger.error(f"Revoking permissions for client_id: {client_id} due to status: {data['status']}")
            revoke_client_permissions(client_id)

        return jsonify({"message": "Mise à jour de l'abonnement réussie"}), 200
    except ValueError as e:
        app.logger.error(f"Date parsing error: {str(e)}")
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD HH:MM:SS"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
            history = image_manager.get_user_history(wp_user_id, "generated")
            return render_template('generate-image.html', history=history, LOGIN_URL=LOGIN_URL)

        # POST request - generate image
        if request.method == 'POST':
            prompt = request.form.get('prompt')
            model = request.form.get('model', 'dall-e')  # Default to DALL-E if not specified
            app.logger.error(f"Processing request - Model: {model}, Prompt: {prompt}")

            if not prompt:
                return jsonify({"message": "Create what inspires you!"}), 400

            try:
                # Utiliser le gestionnaire de modèles
                result = ai_manager.generate_image_sync(model, prompt)
                app.logger.error(f"Generation result: {result}")

                if result['success']:
                    if model == 'midjourney':
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

                            return jsonify({
                                'success': True,
                                'image_key': image_key,
                                'image_url': f"/image/{image_key}"
                            })
                        else:
                            return jsonify({'error': 'Failed to download image'}), 500
                else:
                    return jsonify({'error': result.get('error', 'Failed to generate image')}), 500

            except openai.OpenAIError as e:
                error_message = str(e)
                if "billing_hard_limit_reached" in error_message:
                    error_message = "Billing limit reached. Please check your OpenAI account."
                return jsonify({'error': error_message}), 500
            except requests.RequestException as e:
                app.logger.error(f"Request error: {str(e)}")
                return jsonify({'error': 'Failed to connect to image service'}), 500

        return jsonify({"error": "Invalid request method"}), 400

    except Exception as e:
        app.logger.error(f"Unexpected error in generate_image: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

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

@app.route('/image/<image_key>')
def serve_image(image_key):
    image_data = image_manager.get_image(image_key)
    if not image_data:
        return "Image not found", 404
    return send_file(
        BytesIO(image_data),
        mimetype='image/png'
    )

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

    # Vérifier le task_id
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

        # Récupérer les données de la tâche
        task_data = redis_client.hgetall(metadata_key)
        group_data = redis_client.get(group_key)

        app.logger.error(f"Task data from Redis: {task_data}")
        app.logger.error(f"Group data from Redis: {group_data}")

        if not task_data:
            return jsonify({
                "success": False,
                "error": "Task not found"
            }), 404

        # Décoder les données de la tâche
        status = task_data.get(b'status', b'processing').decode('utf-8')
        prompt = task_data.get(b'prompt', b'').decode('utf-8')

        # Préparer la réponse de base
        response = {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": status,
                "prompt": prompt,
                "timestamp": task_data.get(b'timestamp', b'').decode('utf-8')
            }
        }

        # Ajouter les données du groupe si disponibles
        if group_data:
            group_info = json.loads(group_data)
            response["data"].update({
                "initial_grid": group_info.get('initial_grid'),
                "images": group_info.get('images', []),
                "group_status": group_info.get('status', 'pending')
            })

        # Si le status est une erreur, inclure le message d'erreur
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

        # Validation des données reçues
        required_fields = ['task_id', 'image_url', 'variation_number', 'type']
        if not all(field in data for field in required_fields):
            app.logger.error("Missing required fields in callback data")
            return jsonify({"error": "Missing required fields"}), 400

        task_id = data['task_id']
        image_url = data['image_url']
        variation_number = data['variation_number']
        type = data['type']  # 'initial' ou 'upscale'

        # Récupérer les clés Redis
        metadata_key = f"midjourney_task:{task_id}"
        group_key = f"midjourney_group:{task_id}"

        # Vérifier si la tâche existe
        if not redis_client.exists(metadata_key):
            app.logger.error(f"Task {task_id} not found")
            return jsonify({"error": "Task not found"}), 404

        # Utiliser une pipeline Redis pour les opérations atomiques
        pipe = redis_client.pipeline()

        if type == 'initial':
            # Stocker la grille initiale
            group_data = {
                'task_id': task_id,
                'prompt': redis_client.hget(metadata_key, b'prompt').decode('utf-8'),
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

        elif type == 'upscale':
            # Récupérer les données du groupe existant
            group_data = redis_client.get(group_key)
            if group_data:
                group_info = json.loads(group_data)

                # Ajouter la nouvelle image upscale
                new_image = {
                    'url': image_url,
                    'variation_number': variation_number,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'upscale'
                }

                group_info['images'].append(new_image)

                # Mettre à jour le statut si toutes les images sont reçues
                if len(group_info['images']) == 4:
                    group_info['status'] = 'complete'
                    pipe.hset(metadata_key, 'status', b'completed')

                # Sauvegarder les modifications du groupe
                pipe.setex(
                    group_key,
                    TEMP_STORAGE_DURATION,
                    json.dumps(group_info)
                )

        # Exécuter toutes les opérations Redis
        pipe.execute()

        return jsonify({
            "success": True,
            "task_id": task_id,
            "type": type,
            "status": "complete" if type == 'upscale' and len(group_info['images']) == 4 else "processing"
        })

    except Exception as e:
        app.logger.error(f"Error in callback: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

@app.after_request
def after_request(response):
    app.logger.error("=== CORS Headers Debug ===")
    app.logger.error(f"Response Headers: {dict(response.headers)}")
    return response


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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5432)