import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, send_file, abort, redirect
from io import BytesIO
from PIL import Image
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

from config import redis_client, TEMP_STORAGE_DURATION, PERMANENT_STORAGE_DURATION


load_dotenv(dotenv_path='.env')

openai.api_key = os.getenv("openai.api_key")
SECRET_KEY = os.getenv('SECRET_KEY')
FIXED_TOKEN = os.getenv('FIXED_TOKEN')
LOGIN_URL = "https://aitheway.com/login/"

app = Flask(__name__)

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
                metadata['created_at'] = datetime.now().isoformat()
                pipe.hmset(metadata_key, metadata)
                pipe.expire(metadata_key, TEMP_STORAGE_DURATION)

            # Ajouter à l'index utilisateur
            user_index_key = f"user:{user_id}:images"
            pipe.lpush(user_index_key, key)

            pipe.ltrim(user_index_key, 0, 99)  # Garder les 100 dernières images

            # Exécuter toutes les opérations
            pipe.execute()

            app.logger.error(f"Image stored with key: {key}")
            app.logger.error(f"Metadata stored at: {metadata_key if metadata else 'No metadata'}")

            return key

        except Exception as e:
            app.logger.error(f"Redis storage error: {str(e)}")
            raise


    def get_user_history(self, user_id, history_type="enhanced"):
        """Récupère l'historique des images d'un utilisateur"""
        app.logger.error(f"Fetching history for user {user_id} with type {history_type}")
        if history_type == "enhanced":
            pattern = f"temp:image:{user_id}:*"

            app.logger.error(f"Searching with pattern: {pattern}")
            app.logger.error(f"Found keys: {self.redis.keys(pattern)}")

            images = []

            for key in self.redis.keys(pattern):
                app.logger.error(f"Redis keys found for pattern {pattern}: {self.redis.keys(pattern)}")
                metadata = self.redis.hgetall(f"{key}:meta")
                if metadata.get(b'type') == b'enhanced':
                    original_key = metadata.get(b'original_key')
                    try:
                        images.append({
                            'enhanced_key': key.decode('utf-8'),
                            'original_key': original_key.decode('utf-8') if original_key else None,
                            'timestamp': metadata.get(b'timestamp', b'').decode('utf-8'),
                            'enhanced_url': f"/image/{key.decode('utf-8')}",
                            'original_url': f"/image/{original_key.decode('utf-8')}" if original_key else None
                        })
                    except Exception as e:
                        app.logger.error(f"Error processing enhanced image metadata: {e}")


        elif history_type == "generated":
            pattern = f"img:temp:image:{user_id}:*"
            images = []

            for key in self.redis.keys(pattern):
                app.logger.error(f"Redis keys found for pattern {pattern}: {self.redis.keys(pattern)}")
                metadata = self.redis.hgetall(f"{key}:meta")
                app.logger.error(f"Metadata for key {key}: {metadata}")

                if metadata.get(b'type') == b'generated':
                    try:
                        # Utiliser .get() avec une valeur par défaut pour éviter les erreurs
                        decoded_prompt = metadata.get(b'prompt', b'').decode('utf-8')
                        decoded_timestamp = metadata.get(b'timestamp', b'').decode('utf-8')

                        # Charger les paramètres de manière sécurisée
                        try:
                            parameters = json.loads(metadata.get(b'parameters', b'{}').decode('utf-8'))
                        except (json.JSONDecodeError, TypeError):
                            parameters = {}

                        images.append({
                            'key': key.decode('utf-8'),
                            'prompt': decoded_prompt,
                            'timestamp': decoded_timestamp,
                            'url': f"/image/{key.decode('utf-8')}",
                            'parameters': parameters,
                        })
                    except Exception as e:
                        app.logger.error(f"Error processing generated image metadata: {e}")
        else:
            app.logger.error(f"Invalid history type requested: {history_type}")
            # Gérer d'autres types d'historique si nécessaire
            return []

        sorted_images = sorted(images, key=lambda x: x['timestamp'], reverse=True)
        app.logger.error(f"Sorted images: {sorted_images}")
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

def cleanup_expired_images():
    """Nettoie les images expirées de Redis"""
    pattern = "temp:image:*"
    for key in redis_client.keys(pattern):
        if not redis_client.ttl(key):  # Si pas de TTL ou expiré
            redis_client.delete(key)

image_manager = ImageManager(redis_client)

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
            return f(None)
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

            return f(wp_user_id)

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
            return render_template('generate-image.html', history=history)

        # POST request - generate image
        if request.method == 'POST':
            prompt = request.form.get('prompt')
            if not prompt:
                return jsonify({"message": "Create what inspires you!"}), 400

            try:
                response = openai.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )

                image_url = response.data[0].url
                image_response = requests.get(image_url)

                if image_response.status_code == 200:
                    # Préparer les métadonnées
                    metadata = {
                        'type': 'generated',
                        'prompt': prompt,
                        'timestamp': datetime.now().isoformat(),
                        'parameters': json.dumps({
                            'model': 'dall-e-3',
                            'size': '1024x1024'
                        })
                    }

                    # Stockage de l'image avec ses métadonnées
                    image_key = image_manager.store_temp_image(
                        wp_user_id,
                        image_response.content,
                        metadata
                    )
                    app.logger.error(f"Storing image with key: {image_key}")
                    app.logger.error(f"Metadata stored: {metadata}")

                    return jsonify({
                        'success': True,
                        'image_key': image_key,
                        'image_url': f"/image/{image_key}"
                    })
                else:
                    return jsonify({'error': 'Failed to download image'}), 500

            except openai.OpenAIError as e:
                error_message = str(e)
                if "billing_hard_limit_reached" in error_message:
                    error_message = "Billing limit reached. Please check your OpenAI account."
                return jsonify({'error': error_message}), 500

        # If somehow we get here without returning
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

                enhanced_url = enhance_image_quality(BytesIO(image_data))
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

        return jsonify({"error": "Method not allowed"}), 405

    except Exception as e:
        app.logger.error(f"Unexpected error in /upload_enhance: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def enhance_image_quality(file_path):
    try:
        if not deep_ai_api_key:
            app.logger.error("DeepAI API key is not set")
            return None

        with open(file_path, 'rb') as image_file:
            app.logger.error("Attempting to contact DeepAI API...")
            response = requests.post(
                'https://api.deepai.org/api/waifu2x',
                files={'image': image_file},
                headers={'api-key': deep_ai_api_key},
                timeout=35
            )
            app.logger.error(f"DeepAI API Status Code: {response.status_code}")
            app.logger.error(f"DeepAI API Response Headers: {response.headers}")

            if response.status_code != 200:
                app.logger.error(f"DeepAI API Error: Status {response.status_code}")
                app.logger.error(f"Response content: {response.text}")
                return None

            result = response.json()
            app.logger.debug(f"Response from DeepAI: {result}")
            if 'output_url' not in result:
                app.logger.error(f"No output URL in response: {result}")
                return None
            return result.get('output_url')

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error contacting DeepAI: {str(e)}")
        return None
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


@app.route('/api/chat/history/generated')
@token_required
def get_chat_history(wp_user_id):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    filter_type = request.args.get('type')  # optionnel

    try:
        history = chat_manager.get_user_chat_history(
            wp_user_id,
            page=page,
            per_page=per_page,
            filter_type=filter_type
        )

        return jsonify({
            'success': True,
            'data': history
        })

    except Exception as e:
        app.logger.error(f"Error fetching chat history: {str(e)}")
        app.logger.error(f"Error in /api/chat/history/generated: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch chat history'
        }), 500


def get_generated_history(wp_user_id):
    """Récupère l'historique des images générées pour un utilisateur."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    app.logger.error("=== Fetching Generated History ===")
    app.logger.error(f"User ID: {wp_user_id}")
    app.logger.error(f"Page: {page}, Per Page: {per_page}")

    try:
        # Appel à la méthode pour récupérer l'historique
        history = image_manager.get_user_history(
            wp_user_id,
            history_type="generated"
        )

        app.logger.error(f"Fetched history: {history}")

        # Paginer les résultats
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_history = history[start_idx:end_idx]

        has_more = len(history) > end_idx
        app.logger.error(f"Paginated History: {paginated_history}")
        app.logger.error(f"Has More: {has_more}")

        return jsonify({
            'success': True,
            'items': paginated_history,
            'has_more': has_more
        }), 200

    except Exception as e:
        app.logger.error(f"Error fetching generated history for user {wp_user_id}: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to fetch generated history'}), 500

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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5432)