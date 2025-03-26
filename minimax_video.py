import requests
import json
import time
import os
import logging
import traceback
from datetime import datetime
from flask import current_app as app
from abc import ABC


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

    def process_image_for_minimax(image_data, max_size_mb=3):
        """
        Process an image to ensure it's suitable for MiniMax API:
        1. Resize if too large
        2. Convert to appropriate format
        3. Encode to base64 with proper MIME type

        Args:
            image_data: Raw image bytes
            max_size_mb: Maximum size in MB for the base64 encoded result

        Returns:
            Base64 encoded image string with MIME prefix
        """
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            import math

            # Load the image
            img = BytesIO(image_data)
            image = Image.open(img)

            # Get original dimensions and format
            width, height = image.size
            original_format = image.format or "PNG"

            # Calculate target size (aim for 75% of max to be safe)
            target_size_bytes = int(max_size_mb * 0.75 * 1024 * 1024)

            # Start with original size
            new_width, new_height = width, height
            output_quality = 90

            # Try different compression levels if needed
            for attempt in range(3):
                # Create output buffer
                output = BytesIO()

                # Save with current parameters
                if original_format == "PNG":
                    # For PNGs, use optimize and reduce colors if needed
                    if attempt == 0:
                        image.save(output, format="PNG", optimize=True)
                    elif attempt == 1:
                        # Convert to RGB if it's RGBA
                        if image.mode == 'RGBA':
                            rgb_img = Image.new('RGB', image.size, (255, 255, 255))
                            rgb_img.paste(image, mask=image.split()[3])
                            rgb_img.save(output, format="JPEG", quality=output_quality, optimize=True)
                        else:
                            image.save(output, format="JPEG", quality=output_quality, optimize=True)
                    else:
                        # Resize to 75% dimensions (about 56% of original area)
                        resize_factor = 0.75
                        new_width = int(width * resize_factor)
                        new_height = int(height * resize_factor)
                        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
                        resized_img.save(output, format="JPEG", quality=output_quality, optimize=True)
                else:
                    # For JPEGs, just adjust quality
                    if attempt == 0:
                        image.save(output, format="JPEG", quality=output_quality, optimize=True)
                    elif attempt == 1:
                        output_quality = 70
                        image.save(output, format="JPEG", quality=output_quality, optimize=True)
                    else:
                        # Resize to 75% dimensions
                        resize_factor = 0.75
                        new_width = int(width * resize_factor)
                        new_height = int(height * resize_factor)
                        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
                        resized_img.save(output, format="JPEG", quality=output_quality, optimize=True)

                # Check if we've achieved target size
                output.seek(0)
                data = output.getvalue()
                encoded = base64.b64encode(data).decode('utf-8')

                app.logger.error(
                    f"Processed image attempt {attempt + 1}: format={'JPEG' if attempt > 0 else original_format}, "
                    f"quality={output_quality if attempt > 0 else 'default'}, "
                    f"size={len(encoded) / 1024 / 1024:.2f}MB, "
                    f"dimensions={new_width}x{new_height}")

                if len(encoded) <= target_size_bytes:
                    # Success - return with proper MIME type
                    mime_type = "image/jpeg" if attempt > 0 else f"image/{original_format.lower()}"
                    return f"data:{mime_type};base64,{encoded}"

                # If still too large on final attempt, make drastic reduction
                if attempt == 2:
                    # Calculate how much we need to reduce
                    size_ratio = math.sqrt(target_size_bytes / len(encoded))
                    final_width = int(new_width * size_ratio)
                    final_height = int(new_height * size_ratio)

                    # Ensure minimum dimensions (MiniMax might require at least 512x512)
                    final_width = max(512, final_width)
                    final_height = max(512, final_height)

                    output = BytesIO()
                    final_img = image.resize((final_width, final_height), Image.LANCZOS)
                    final_img.save(output, format="JPEG", quality=60, optimize=True)
                    output.seek(0)
                    data = output.getvalue()
                    encoded = base64.b64encode(data).decode('utf-8')

                    app.logger.error(f"Final resize: dimensions={final_width}x{final_height}, "
                                     f"size={len(encoded) / 1024 / 1024:.2f}MB")

                    return f"data:image/jpeg;base64,{encoded}"

            # Shouldn't reach here due to final resize above
            return f"data:image/jpeg;base64,{encoded}"

        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            # Return original image data with default MIME type as fallback
            import base64
            encoded = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{encoded}"

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