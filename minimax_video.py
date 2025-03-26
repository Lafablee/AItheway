import requests
import json
import time
import os
import logging
import traceback
from datetime import datetime
from flask import current_app as app
from abc import ABC


def process_image_for_minimax(image_data, max_size_mb=5):
    """
    Optimise l'image pour l'API MiniMax en réduisant sa taille si nécessaire

    Args:
        image_data: Données binaires de l'image
        max_size_mb: Taille maximale en MB pour l'image encodée en base64

    Returns:
        String base64 encodée avec préfixe MIME
    """
    try:
        from PIL import Image
        import base64
        from io import BytesIO

        # Charger l'image
        img = Image.open(BytesIO(image_data))
        width, height = img.size

        # Vérifier le ratio (doit être entre 2:5 et 5:2)
        ratio = width / height
        if ratio < 0.4 or ratio > 2.5:
            app.logger.error(f"Image ratio {ratio} is outside allowed range (0.4-2.5)")

        # Réduire progressivement jusqu'à obtenir une taille acceptable
        quality = 90
        max_size = max_size_mb * 1024 * 1024  # Convertir en bytes
        output = BytesIO()

        # Essayer avec PNG d'abord
        img.save(output, format="PNG", optimize=True)
        output.seek(0)

        if len(output.getvalue()) > max_size:
            # Passer à JPEG avec qualité progressive
            for quality in [95, 85, 75, 65]:
                output = BytesIO()
                if img.mode == 'RGBA':
                    # Convertir RGBA en RGB pour JPEG
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    rgb_img.save(output, format="JPEG", quality=quality, optimize=True)
                else:
                    img.save(output, format="JPEG", quality=quality, optimize=True)

                output.seek(0)
                if len(output.getvalue()) <= max_size:
                    break

            # Si toujours trop grand, redimensionner
            if len(output.getvalue()) > max_size:
                scale_factor = 0.8
                while len(output.getvalue()) > max_size:
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)

                    # S'assurer que le côté le plus court reste > 300px
                    if min(new_width, new_height) <= 300:
                        break

                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    output = BytesIO()
                    resized.save(output, format="JPEG", quality=quality, optimize=True)
                    output.seek(0)

                    if len(output.getvalue()) <= max_size:
                        break

                    scale_factor *= 0.8

        # Encoder en base64
        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        app.logger.error(
            f"Image processed: original {len(image_data) / 1024 / 1024:.2f}MB, encoded {len(encoded_image) / 1024 / 1024:.2f}MB")

        # Déterminer le type MIME
        mime_type = "image/jpeg" if quality < 100 else "image/png"
        return f"data:{mime_type};base64,{encoded_image}"

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None

class MiniMaxVideoGenerator(ABC):
    """
    Classe pour gérer la génération de vidéos via l'API MiniMax.
    """

    def __init__(self, api_key, default_model="T2V-01-Director"):
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = "https://api.minimaxi.chat/v1"
        self.group_id = os.getenv("MINIMAX_GROUP_ID", "")

    def generate(self, prompt, model=None, additional_params=None):
        """Méthode principale pour générer une vidéo de manière synchrone (attend le résultat)"""
        try:
            task_id = self.create_generation_task(prompt, model, additional_params)
            if not task_id:
                return {"success": False, "error": "Failed to create generation task"}

            # Attendre que la tâche soit terminée (avec timeout)
            status, file_id = self.wait_for_completion(task_id)

            if status == "Success" and file_id:
                # Récupérer l'URL de téléchargement
                download_url = self.get_download_url(file_id)
                if download_url:
                    return {
                        "success": True,
                        "task_id": task_id,
                        "file_id": file_id,
                        "download_url": download_url,
                        "status": status
                    }

            return {
                "success": False,
                "task_id": task_id,
                "status": status,
                "error": "Failed to generate video or retrieve download URL"
            }

        except Exception as e:
            app.logger.error(f"Error in MiniMaxVideoGenerator.generate: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_async(self, prompt, model=None, additional_params=None):
        """Crée une tâche asynchrone et retourne immédiatement le task_id"""
        try:
            # Ne pas remplacer le modèle s'il est déjà spécifié
            if model:
                app.logger.error(f"Using specified model: {model}")
            else:
                # Si first_frame_image est présent mais que le modèle n'est pas spécifié
                if additional_params and "first_frame_image" in additional_params and not model:
                    model = "I2V-01-Director"
                    app.logger.error(f"Auto-selecting I2V-01-Director model for image")
                else:
                    model = self.default_model
                    app.logger.error(f"Using default model: {model}")

            task_id = self.create_generation_task(prompt, model, additional_params)
            if task_id:
                return {"success": True, "task_id": task_id, "status": "processing"}
            return {"success": False, "error": "Failed to create generation task"}
        except Exception as e:
            app.logger.error(f"Error in MiniMaxVideoGenerator.generate_async: {str(e)}")
            app.logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def create_generation_task(self, prompt, model=None, additional_params=None):
        """Creates a video generation task and returns the task_id with improved error handling"""
        headers = {
            'authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Respect le modèle spécifié, ou utilise I2V si image présente sans modèle spécifié
        if model:
            # Utiliser le modèle spécifié par l'appelant - ne pas le remplacer
            app.logger.error(f"Using specified model: {model}")
        elif additional_params and "first_frame_image" in additional_params:
            model = "I2V-01-Director"
            app.logger.error(f"Auto-selecting I2V-01-Director model for image")
        else:
            model = self.default_model
            app.logger.error(f"Using default model: {model}")


        # Ensure we're using the correct model based on whether we have an image
        if additional_params and "first_frame_image" in additional_params:
            model = "I2V-01-Director"  # Force Image-to-Video model when image is present
            app.logger.error(f"Using I2V-01-Director model because reference image is present")
        else:
            model = model or self.default_model

        app.logger.error(f"Creating task with model: {model}")

        payload = {
            "model": model,
            "prompt": prompt,
        }

        # Add additional parameters
        if additional_params:
            if isinstance(additional_params, str):
                try:
                    additional_params = json.loads(additional_params)
                except json.JSONDecodeError:
                    app.logger.error(f"Failed to parse additional_params JSON: {additional_params}")
                    additional_params = {}

            # Handle image reference with size check
            if "first_frame_image" in additional_params:
                base64_image = additional_params["first_frame_image"]

                # Check the size of the base64 image
                image_size_mb = len(base64_image) / 1024 / 1024
                app.logger.error(f"Base64 image size: {image_size_mb:.2f} MB")

                # Warn if image is too large
                if image_size_mb > 10:
                    app.logger.error(f"WARNING: Image is very large ({image_size_mb:.2f} MB), may exceed API limits")

                # Ensure image has proper MIME prefix
                if not base64_image.startswith("data:image"):
                    try:
                        clean_base64 = base64_image.strip().replace('\n', '').replace('\r', '')
                        mime_type = "image/png"
                        base64_image = f"data:{mime_type};base64,{clean_base64}"
                    except Exception as e:
                        app.logger.error(f"Error formatting base64 image: {str(e)}")

                payload["first_frame_image"] = base64_image

            # Handle other parameters
            for param, value in additional_params.items():
                if param != "first_frame_image":  # Skip the image we already processed
                    if param == "subject_reference" or param == "prompt_optimizer":
                        # Convert string booleans to actual booleans
                        if isinstance(value, str):
                            payload[param] = value.lower() == 'true'
                        else:
                            payload[param] = bool(value)
                    else:
                        payload[param] = value

                    app.logger.error(f"Adding parameter {param}: {payload[param]}")

        # Log payload size
        payload_size = len(json.dumps(payload))
        app.logger.error(f"Final payload size: {payload_size / 1024 / 1024:.2f} MB")

        try:
            # Use json.dumps with ensure_ascii=False to preserve characters
            json_payload = json.dumps(payload, ensure_ascii=False)

            app.logger.error(f"Sending request to MiniMax API")
            response = requests.post(
                f"{self.base_url}/video_generation",
                headers=headers,
                data=json_payload
            )

            # Check for HTTP errors first
            if response.status_code != 200:
                app.logger.error(f"HTTP error: {response.status_code} - {response.text}")
                return None

            # Process the JSON response
            response_data = response.json()
            app.logger.error(f"MiniMax API response: {response_data}")

            # Check for API-level errors
            if response_data.get("base_resp", {}).get("status_code") != 0:
                error_code = response_data.get("base_resp", {}).get("status_code")
                error_msg = response_data.get("base_resp", {}).get("status_msg")
                app.logger.error(f"API error {error_code}: {error_msg}")

                # Provide more helpful error messages for common issues
                if error_code == 2013:  # invalid params
                    if payload_size > 20 * 1024 * 1024:  # If payload > 20MB
                        app.logger.error("The request payload is too large. Try using a smaller image.")
                    elif "first_frame_image" in payload:
                        app.logger.error("The image format may be unsupported. Try using a JPEG or PNG image.")
                    else:
                        app.logger.error("Invalid parameters. Check your prompt and other settings.")

                return None

            # Check if task_id is present
            if response_data.get("task_id"):
                app.logger.error(f"Successfully created task with ID: {response_data.get('task_id')}")
                return response_data.get("task_id")
            else:
                app.logger.error("No task_id in response")
                return None

        except Exception as e:
            app.logger.error(f"Exception in create_generation_task: {str(e)}")
            app.logger.error(traceback.format_exc())
            return None

    def check_generation_status(self, task_id):
        """Vérifie le statut d'une tâche de génération"""
        headers = {
            'authorization': f'Bearer {self.api_key}',
            'content-type': 'application/json',
        }

        try:
            response = requests.get(
                f"{self.base_url}/query/video_generation?task_id={task_id}",
                headers=headers
            )

            response_data = response.json()
            app.logger.info(f"MiniMax status check response: {response_data}")

            status = response_data.get("status", "")
            file_id = response_data.get("file_id", "")

            return {
                "success": True,
                "task_id": task_id,
                "status": status,
                "file_id": file_id
            }

        except Exception as e:
            app.logger.error(f"Exception in check_generation_status: {str(e)}")
            return {
                "success": False,
                "task_id": task_id,
                "status": "Error",
                "error": str(e)
            }

    def get_download_url(self, file_id):
        """Récupère l'URL de téléchargement d'un fichier"""
        headers = {
            'authority': 'api.minimaxi.chat',
            'content-type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            url = f'{self.base_url}/files/retrieve?GroupId={self.group_id}&file_id={file_id}'
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                app.logger.info(f"File retrieval response: {response_data}")

                if response_data.get("file") and response_data["file"].get("download_url"):
                    return response_data["file"]["download_url"]

            app.logger.error(f"Error retrieving file download URL: {response.text}")
            return None

        except Exception as e:
            app.logger.error(f"Exception in get_download_url: {str(e)}")
            return None

    def wait_for_completion(self, task_id, max_attempts=30, delay=10):
        """Attend que la tâche soit terminée, avec un nombre maximal de tentatives"""
        for attempt in range(max_attempts):
            app.logger.info(f"Checking status of task {task_id}, attempt {attempt + 1}/{max_attempts}")

            status_response = self.check_generation_status(task_id)
            status = status_response.get("status", "")
            file_id = status_response.get("file_id", "")

            if status == "Success" and file_id:
                app.logger.info(f"Task {task_id} completed successfully with file_id {file_id}")
                return status, file_id

            if status == "Fail":
                app.logger.error(f"Task {task_id} failed")
                return status, None

            app.logger.info(f"Task {task_id} is still {status}, waiting {delay} seconds...")
            time.sleep(delay)

        app.logger.error(f"Timeout waiting for task {task_id} to complete")
        return "Timeout", None