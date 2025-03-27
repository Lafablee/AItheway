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
    Optimise l'image pour l'API MiniMax en s'assurant qu'elle respecte tous les critères:
    - Format: JPG, JPEG ou PNG
    - Ratio d'aspect: entre 2:5 et 5:2 (donc entre 0.4 et 2.5)
    - Dimensions minimales: côté le plus court > 300 pixels
    - Taille du fichier: < 20MB

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
        import math

        # Charger l'image
        img_buffer = BytesIO(image_data)
        img = Image.open(img_buffer)

        # 1. Vérifier et ajuster le format
        original_format = img.format
        app.logger.error(f"Original image format: {original_format}")

        if original_format not in ['JPEG', 'JPG', 'PNG']:
            app.logger.error(f"Converting {original_format} to PNG")
            # On va convertir en PNG par défaut
            if img.mode == 'RGBA':
                # Garder le canal alpha pour PNG
                new_format = 'PNG'
            else:
                # Sinon JPEG est plus petit
                new_format = 'JPEG'
        else:
            new_format = original_format

        # 2. Vérifier et ajuster le ratio d'aspect (entre 0.4 et 2.5)
        width, height = img.size
        app.logger.error(f"Original dimensions: {width}x{height}")

        ratio = width / height
        app.logger.error(f"Original aspect ratio: {ratio:.2f}")

        if ratio < 0.4 or ratio > 2.5:
            app.logger.error(f"Aspect ratio {ratio:.2f} is outside allowed range (0.4-2.5), adjusting...")

            # Recadrer l'image pour obtenir un ratio acceptable
            if ratio < 0.4:  # trop étroit
                new_height = int(width / 0.4)
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))
            elif ratio > 2.5:  # trop large
                new_width = int(height * 2.5)
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))

            # Mettre à jour les dimensions après recadrage
            width, height = img.size
            ratio = width / height
            app.logger.error(f"Adjusted dimensions: {width}x{height}, new ratio: {ratio:.2f}")

        # 3. Vérifier les dimensions minimales (côté le plus court > 300px)
        min_side = min(width, height)
        if min_side < 300:
            app.logger.error(f"Minimum dimension {min_side}px is below 300px, resizing...")

            # Redimensionner tout en préservant le ratio
            scale_factor = 300 / min_side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Mettre à jour les dimensions après redimensionnement
            width, height = img.size
            app.logger.error(f"Resized to: {width}x{height}")

        # 4. Ajuster la qualité pour respecter la limite de taille
        max_size_bytes = int(max_size_mb * 0.75 * 1024 * 1024)  # 75% de la limite pour être sûr

        # Essayer d'abord avec qualité élevée
        quality = 95
        output = BytesIO()

        if new_format == 'PNG':
            # Options d'optimisation pour PNG
            img.save(output, format="PNG", optimize=True, compress_level=9)
        else:
            # Pour JPEG, utiliser la compression par qualité
            if img.mode == 'RGBA':
                # Convertir RGBA en RGB pour JPEG
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                rgb_img.save(output, format="JPEG", quality=quality, optimize=True)
            else:
                img.save(output, format="JPEG", quality=quality, optimize=True)

        output.seek(0)
        data = output.getvalue()

        # Si l'image est trop grande, réessayer avec différentes qualités et tailles
        if len(data) > max_size_bytes:
            app.logger.error(f"Image is too large: {len(data) / 1024 / 1024:.2f}MB, optimizing...")

            # Essayer d'abord de réduire la qualité (pour JPEG)
            if new_format == 'JPEG':
                for q in [85, 75, 65]:
                    output = BytesIO()
                    img.save(output, format="JPEG", quality=q, optimize=True)
                    output.seek(0)
                    data = output.getvalue()

                    if len(data) <= max_size_bytes:
                        app.logger.error(f"Reduced quality to {q}, new size: {len(data) / 1024 / 1024:.2f}MB")
                        break

            # Si toujours trop grand, réduire les dimensions
            if len(data) > max_size_bytes:
                app.logger.error("Still too large, reducing dimensions...")
                scale = 0.9  # Réduire par étapes de 10%

                while len(data) > max_size_bytes and scale > 0.1:
                    new_width = int(width * scale)
                    new_height = int(height * scale)

                    # S'assurer que les dimensions minimales sont respectées
                    if min(new_width, new_height) < 300:
                        break

                    # Redimensionner
                    resized = img.resize((new_width, new_height), Image.LANCZOS)

                    # Enregistrer avec la qualité actuelle
                    output = BytesIO()
                    if new_format == 'PNG':
                        resized.save(output, format="PNG", optimize=True, compress_level=9)
                    else:
                        resized.save(output, format="JPEG", quality=quality, optimize=True)

                    output.seek(0)
                    data = output.getvalue()

                    app.logger.error(f"Resized to {new_width}x{new_height}, new size: {len(data) / 1024 / 1024:.2f}MB")

                    if len(data) <= max_size_bytes:
                        break

                    scale *= 0.9

                # Si toujours trop grand après toutes les optimisations
                if len(data) > max_size_bytes:
                    app.logger.error("WARNING: Still exceeding size limit after optimizations")
                    # Forcer la conversion en JPEG avec qualité minimale acceptable
                    if new_format != 'JPEG':
                        if img.mode == 'RGBA':
                            # Convertir RGBA en RGB
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[3])
                            img = rgb_img

                        new_format = 'JPEG'
                        quality = 60

                        output = BytesIO()
                        img.save(output, format="JPEG", quality=quality, optimize=True)
                        output.seek(0)
                        data = output.getvalue()
                        app.logger.error(f"Forced conversion to JPEG, quality 60: {len(data) / 1024 / 1024:.2f}MB")

        # Encoder en base64
        encoded = base64.b64encode(data).decode('utf-8')

        # Log des informations finales
        app.logger.error(
            f"Final image: format={new_format}, dimensions={img.size}, size={len(data) / 1024 / 1024:.2f}MB, encoded={len(encoded) / 1024 / 1024:.2f}MB")

        # Ajouter le préfixe MIME approprié
        mime_type = "image/jpeg" if new_format == 'JPEG' else "image/png"
        return f"data:{mime_type};base64,{encoded}"

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
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
        """Crée une tâche de génération et retourne le task_id avec traitement d'erreur amélioré"""
        headers = {
            'authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Assurer que le modèle est correctement choisi
        if additional_params and "first_frame_image" in additional_params:
            model = "I2V-01-Director"  # Forcer I2V-01-Director quand une image est présente
            app.logger.error(f"Using I2V-01-Director model because reference image is present")
        else:
            model = model or self.default_model

        app.logger.error(f"Creating task with model: {model}")

        # Construction du payload de base
        payload = {
            "model": model,
            "prompt": prompt.strip(),  # Supprimer les espaces inutiles
        }

        # Ajouter les paramètres additionnels validés
        if additional_params:
            if isinstance(additional_params, str):
                try:
                    additional_params = json.loads(additional_params)
                except json.JSONDecodeError:
                    app.logger.error(f"Failed to parse additional_params JSON: {additional_params}")
                    additional_params = {}

            # Traitement de l'image de référence
            if "first_frame_image" in additional_params:
                base64_image = additional_params["first_frame_image"]

                # Vérifier si l'image est correctement formatée
                if not base64_image.startswith("data:image/"):
                    app.logger.error("Image missing MIME prefix, adding it")
                    base64_image = f"data:image/png;base64,{base64_image}"

                # Vérifier la taille de l'image encodée
                image_size_mb = len(base64_image) / 1024 / 1024
                app.logger.error(f"Base64 image size: {image_size_mb:.2f} MB")

                payload["first_frame_image"] = base64_image

            # Nettoyage et validation des autres paramètres
            supported_params = ["prompt_optimizer", "subject_reference", "callback_url"]
            for param in supported_params:
                if param in additional_params:
                    # Conversion en booléen pour les paramètres qui le nécessitent
                    if param in ["prompt_optimizer", "subject_reference"]:
                        if isinstance(additional_params[param], str):
                            payload[param] = additional_params[param].lower() == 'true'
                        else:
                            payload[param] = bool(additional_params[param])
                    else:
                        payload[param] = additional_params[param]

                    app.logger.error(f"Adding parameter {param}: {payload[param]}")

        # Calculer et afficher la taille du payload
        json_str = json.dumps(payload, ensure_ascii=False)
        payload_size = len(json_str) / 1024 / 1024
        app.logger.error(f"Final payload size: {payload_size:.2f} MB")

        try:
            # Envoi de la requête à l'API
            app.logger.error(f"Sending request to MiniMax API")
            response = requests.post(
                f"{self.base_url}/video_generation",
                headers=headers,
                data=json_str
            )

            # Afficher les détails de la réponse pour le débogage
            status_code = response.status_code
            response_text = response.text
            app.logger.error(f"API Response Code: {status_code}")
            app.logger.error(f"API Response Text: {response_text[:500]}")  # Tronqué pour éviter des logs trop longs

            try:
                response_data = response.json()
                app.logger.error(f"MiniMax API response: {response_data}")
            except json.JSONDecodeError:
                app.logger.error("Failed to parse API response as JSON")
                return None

            # Validation de la réponse
            if response.status_code != 200:
                app.logger.error(f"HTTP error: {response.status_code}")
                return None

            # Vérification du code de statut dans la réponse
            base_resp = response_data.get("base_resp", {})
            status_code = base_resp.get("status_code", -1)
            status_msg = base_resp.get("status_msg", "unknown error")

            if status_code != 0:
                app.logger.error(f"API error {status_code}: {status_msg}")

                # Messages d'erreur spécifiques
                if status_code == 2013:  # invalid params
                    # Vérifier quelle partie du payload pose problème
                    if model != "I2V-01-Director" and "first_frame_image" in payload:
                        app.logger.error(f"Image was provided but model {model} doesn't support images!")
                    elif "first_frame_image" in payload:
                        # Problèmes potentiels avec l'image
                        img_info = "MIME prefix: " + payload["first_frame_image"][:30] + "..."
                        app.logger.error(f"Image validation error. {img_info}")
                        app.logger.error("Check image format, aspect ratio, and minimum size requirements")
                    else:
                        app.logger.error(f"Invalid parameter in request: {payload.keys()}")

                return None

            # Récupération du task_id
            task_id = response_data.get("task_id")
            if task_id:
                app.logger.error(f"Successfully created task with ID: {task_id}")
                return task_id
            else:
                app.logger.error("No task_id in response")
                return None

        except Exception as e:
            app.logger.error(f"Exception in create_generation_task: {str(e)}")
            import traceback
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