import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, send_file, abort
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import jwt
from jwt import InvalidTokenError
from functools import wraps
import psycopg2




load_dotenv(dotenv_path='.env')

openai.api_key = os.getenv("openai.api_key")
SECRET_KEY = os.getenv('SECRET_KEY')

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
        SELECT id, email, subscription_level, status FROM clients WHERE wp_user_id = %s
    """, (wp_user_id,))
    client = cursor.fetchone()
    cursor.close()
    conn.close()

    if client:
        return{
            "id": client[0],
            "email": client[1],
            "subscription_level": client[2],
            "status": client[3]
        }
    else:
        return None

def check_client_permission(client_id, action):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT is_allowed FROM permissions WHERE client_id = %s AND action = %s
    """, (client_id, action))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result is not None and result[0]

def verify_jwt_token(token):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        wp_user_id = decoded_token.get("data", {}).get("user", {}).get("id")
        return wp_user_id  # Retourne uniquement wp_user_id si le token est valide
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            abort(401, description="Token missing or invalid")

        token = auth_header.split(" ")[1]
        wp_user_id = verify_jwt_token(token)
        if not wp_user_id:
            abort(403, description="Token is invalid")

        return f(wp_user_id, *args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_format(img_format):
    supported_formats = ['PNG', 'JPEG', 'JPG', 'GIF']
    if img_format not in supported_formats:
        return False, f"Unsupported format: {img_format}"
    if img_format == 'JPG':
        img_format = 'JPEG'
    return True, img_format

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['GET', 'POST'])
@token_required
def generate_image(decoded_token):
    client = None
    # Récupérer l'ID utilisateur depuis le token décodé
    wp_user_id = decoded_token.get("data", {}).get("user", {}).get("id")
    if not client:
        return jsonify({"error": "Client non trouvé"}), 404
    if not check_client_permission(client["id"], "generate_image"):
        return jsonify({"error": "Permission refusée pour générer une image"}), 403


    # Vérifie si le client a la permission d'accéder à cette fonctionnalité
    if request.method == 'GET':
        return render_template('generate-image.html')

    if request.method == 'POST':
        prompt = request.form.get('prompt')

        if not prompt:
            return jsonify({"message": "create what inspires you !"})
        app.logger.info(f"Received prompt: {prompt}")

        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response.data[0].url
            app.logger.info(f"Generated image URL: {image_url}")

            # Télécharger l'image depuis l'URL retournée par OpenAI
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                filename = secure_filename(f"{prompt}.png")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(file_path, "PNG")
                app.logger.info(f"Image saved at {file_path}")
                return jsonify({'image_url': url_for('static', filename=f'uploads/{filename}')})
            else:
                app.logger.error(f"Failed to download image from URL: {image_url}")
                return jsonify({'error': 'Failed to download image'}), 500


        except openai.OpenAIError as e:
            error_message = str(e)
            if "billing_hard_limit_reached" in error_message:
                error_message = "Your billing limit has been reached. Please check your OpenAI account."
            app.logger.error(f"Error generating image: {error_message}")
            return jsonify({'error': error_message})

    return render_template('generate-image.html')

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
def upload_enhance(decoded_token):
    client = None
    # Récupérer l'ID utilisateur depuis le token décodé
    wp_user_id = decoded_token.get("data", {}).get("user", {}).get("id")

    # Vérifie si le client existe et a la permission d'utiliser cette route
    client = get_client_by_wp_user_id(wp_user_id)
    if not client:
        app.logger.error("client unsolved !")
        return jsonify({"error": "Client non trouvé"}), 404

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
