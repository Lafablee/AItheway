import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, send_file, abort, redirect
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import jwt
from jwt import InvalidTokenError
from functools import wraps
import psycopg2
from datetime import datetime, timedelta




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

deep_ai_api_key = os.getenv('DEEPAI_API_KEY')


def get_client_by_wp_user_id(wp_user_id):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, email, subscription_level, status, expiration_date FROM clients WHERE wp_user_id = %s
    """, (wp_user_id,))
    client = cursor.fetchone()
    cursor.close()
    conn.close()
    return {
        "id": client[0],
        "email": client[1],
        "subscription_level": client[2],
        "status": client[3],
        "expiration_date": client[4],
    } if client else None

def add_permission(client_id, action):
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

def check_client_permission(client_id, action):
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
        if is_allowed and ((status == 'active' and now <= expiration_date) or (status == 'cancelled' and now <= expiration_date)):
            return True
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
        token = request.args.get('token')  # Retrieve token from URL query parameter
        app.logger.error('token' + " Checking token from URL parameters")

        if not token:
            app.logger.error("No token found in request")
            auth_header = request.headers.get('Authorization')  # Fallback to Authorization header
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

            if not token:
                return redirect(LOGIN_URL)

            if token == FIXED_TOKEN:
                return f(None, *args, **kwargs)
            try:
                decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                wp_user_id = decoded_token.get("data", {}).get("user", {}).get("id")
                app.logger.info(f"Token decoded successfully for user {wp_user_id}")
                if not wp_user_id:
                    return redirect(LOGIN_URL)

                client = get_client_by_wp_user_id(wp_user_id)
                if not client:
                    return redirect(LOGIN_URL)

                return f(wp_user_id, *args, **kwargs)

            except jwt.InvalidTokenError:
                app.logger.error(f"Invalid token: ")
                abort(403, description="Token is invalid")
                #Next Step : mise en place redirection template error html

    return decorated_function

def add_client(wp_user_id, email, subscription_level, status, expiration_date):
    start_date = datetime.now()
    expiration_date = start_date + timedelta(days=1)

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO clients (wp_user_id, email, subscription_level, status, start_date, expiration_date)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """, (wp_user_id, email, subscription_level, status, datetime.now(), expiration_date))
    client_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return client_id

def update_client_subscription(client_id, subscription_level, status, expiration_date):
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
    conn.commit()
    cursor.close()
    conn.close()


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
        "basic": ["view_content", "generate_image"],
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
        SELECT id FROM clients WHERE status = 'cancelled' AND expiration_date <= %s
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
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("DELETE FROM permissions WHERE client_id = %s", (client_id,))
    conn.commit()
    cursor.close()
    conn.close()

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

@app.route('/sync-membership', methods=['POST'])
@token_required
def sync_membership(wp_user_id):
    #Si le token est fixe, `wp_user_id` sera `None`
    if wp_user_id is None:
        auth_header = request.headers.get('Authorization')
        if auth_header != f'Bearer {FIXED_TOKEN}':
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    wp_user_id = data.get('wp_user_id')
    email = data.get('email')
    subscription_level = data.get('subscription_level')
    status = data.get('status')

    app.logger.info(f"Received subscription sync: Level={subscription_level}, Status={status}")

    if 'expiration_date' not in data or not data['expiration_date']:
        return jsonify({"error": "Missing 'expiration_date' in request data"}), 400

    try:
        expiration_date = datetime.strptime(data['expiration_date'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    # Vérifie si le client existe
    client = get_client_by_wp_user_id(wp_user_id)
    if not client:
        # Si le client n'existe pas, ajoute-le avec les permissions de base
        client_id = add_client(wp_user_id, email, subscription_level, status, expiration_date)
    else:
        client_id = client["id"]
        # Met à jour uniquement si le statut d'abonement change
        if client["subscription_level"] != subscription_level or client["status"] != status or client["expiration_date"] != expiration_date:
            update_client_subscription(client_id, subscription_level, status, expiration_date)

        # Attribue les permissions en fonction du niveau d'abonnement
        if status == 'active':
            assign_permissions(client_id, subscription_level)
        elif status == 'cancelled' and datetime.now() <= expiration_date:
            app.logger.info(f"User {wp_user_id} retains access until expiration date.")
        else:
            app.logger.info(f"No changes for user {wp_user_id}.")

    return jsonify({"message": "Mise à jour de l'abonnement réussie"}), 200


@app.route('/generate_image', methods=['GET', 'POST'])
#@token_required
def generate_image(wp_user_id):
    try:
        client = get_client_by_wp_user_id(wp_user_id)

        # Initial checks
        if not client:
            app.logger.debug(f"User {wp_user_id} tried to connect.")
            return jsonify({"error": "Client not found"}), 404

        if not check_client_permission(client["id"], "generate_image"):
            app.logger.debug(f"User {wp_user_id} permission denied for genreate an image .")
            return jsonify({"error": "Permission denied"}), 403

        # GET request - show form
        if request.method == 'GET':
            return render_template('generate-image.html')

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
                    img = Image.open(BytesIO(image_response.content))
                    filename = secure_filename(f"{prompt}.png")
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    img.save(file_path, "PNG")
                    return jsonify({'image_url': url_for('static', filename=f'uploads/{filename}')})
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
        return jsonify({"error": "Internal server error"})

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
    client = get_client_by_wp_user_id(wp_user_id)
    # vérifie si le client existe dans la base de données

    if not client:
        app.logger.error("client non trouvé !")
        return jsonify({"error": "Client non trouvé"}), 404
    # Vérifie si le client existe et a la permission d'utiliser cette route
    if not check_client_permission(client["id"], "upload_enhance"):
        app.logger.error("Permission refusée pour l'amélioration d'image")
        return jsonify({"error": "Permission refusée pour l'upload et l'amélioration d'image"}), 403

    if request.method == 'GET':
        return render_template('upload-enhance.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.logger.info(f"File saved to {file_path}")

            enhanced_image_url = enhance_image_quality(file_path)
            if enhanced_image_url:
                app.logger.info(f"Enhanced image URL: {enhanced_image_url}")
                return jsonify({'image_url': enhanced_image_url})
            else:
                app.logger.error("Failed to enhance image")
                return jsonify({'error': 'Failed to enhance image'}), 500

        return jsonify({'error': 'Invalid file'}), 400

def enhance_image_quality(file_path):
    with open(file_path, 'rb') as image_file:
        response = requests.post(
            'https://api.deepai.org/api/waifu2x',
            files={'image': image_file},
            headers={'api-key': deep_ai_api_key}
        )
        result = response.json()
        app.logger.debug(f"Response from DeepAI: {result}")
        return result.get('output_url', None)

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
