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
    Traite et valide une image pour l'API MiniMax avec des messages d'erreur clairs pour le frontend.

    Vérifie:
    - Format: uniquement JPG, JPEG ou PNG (rejet strict des autres formats)
    - Ratio d'aspect: entre 2:5 et 5:2 (0.4 à 2.5)
    - Dimensions minimales: côté le plus court >= 300 pixels
    - Taille maximale: < max_size_mb (limité à 20MB par l'API)

    Args:
        image_data: Données binaires de l'image
        max_size_mb: Taille maximale en MB pour l'image encodée en base64

    Returns:
        Un dictionnaire avec:
        - "success": bool - True si le traitement a réussi
        - "image": str - Image encodée en base64 avec préfixe MIME (si success=True)
        - "error_code": str - Code d'erreur standardisé (si success=False)
        - "error_message": str - Message d'erreur détaillé (si success=False)
        - "corrections": list - Liste des corrections appliquées (si success=True)
    """
    try:
        from PIL import Image
        import base64
        from io import BytesIO
        import imghdr  # Pour la détection du type d'image
        import math

        corrections = []

        # Détecter le format réel de l'image (indépendamment de l'extension)
        image_format = imghdr.what(None, h=image_data)

        # Liste des formats autorisés par MiniMax
        allowed_formats = {'jpeg', 'jpg', 'png'}

        # 1. VALIDATION DU FORMAT
        if image_format is None:
            return {
                "success": False,
                "error_code": "INVALID_FORMAT",
                "error_message": "Format d'image non reconnu. Seuls les formats JPG, JPEG et PNG sont acceptés."
            }

        if image_format.lower() not in allowed_formats:
            return {
                "success": False,
                "error_code": "UNSUPPORTED_FORMAT",
                "error_message": f"Format {image_format} non supporté. Seuls les formats JPG, JPEG et PNG sont acceptés."
            }

        # Charger l'image avec PIL
        try:
            img_buffer = BytesIO(image_data)
            img = Image.open(img_buffer)
            original_format = img.format
            app.logger.error(f"Image format: {original_format}, Detected: {image_format}")
        except Exception as e:
            return {
                "success": False,
                "error_code": "CORRUPT_IMAGE",
                "error_message": f"Impossible de traiter l'image. L'image semble corrompue: {str(e)}"
            }

        # 2. VALIDATION DES DIMENSIONS
        width, height = img.size
        min_side = min(width, height)

        if min_side < 300:
            return {
                "success": False,
                "error_code": "IMAGE_TOO_SMALL",
                "error_message": f"L'image est trop petite. La dimension la plus petite est de {min_side}px, mais doit être d'au moins 300px."
            }

        # 3. VALIDATION DU RATIO D'ASPECT
        ratio = width / height
        min_ratio = 0.4  # 2:5
        max_ratio = 2.5  # 5:2

        if ratio < min_ratio or ratio > max_ratio:
            # Calculer le ratio idéal le plus proche pour aider l'utilisateur
            if ratio < min_ratio:
                ideal_height = int(width / min_ratio)
                suggestion = f"Essayez de recadrer la hauteur à {ideal_height}px (actuellement {height}px)"
            else:
                ideal_width = int(height * max_ratio)
                suggestion = f"Essayez de recadrer la largeur à {ideal_width}px (actuellement {width}px)"

            return {
                "success": False,
                "error_code": "INVALID_ASPECT_RATIO",
                "error_message": f"Le ratio d'aspect {ratio:.2f} est en dehors des limites autorisées (entre 0.4 et 2.5). {suggestion}",
                "current_ratio": round(ratio, 2),
                "allowed_min_ratio": min_ratio,
                "allowed_max_ratio": max_ratio
            }

        # 4. CONVERSION SI NÉCESSAIRE POUR ASSURER LA COMPATIBILITÉ
        if original_format not in ['JPEG', 'JPG', 'PNG']:
            new_format = 'PNG' if img.mode == 'RGBA' else 'JPEG'
            corrections.append(f"Format converti de {original_format} à {new_format}")
        else:
            new_format = original_format
            if new_format == 'JPG':
                new_format = 'JPEG'  # Normaliser JPG à JPEG

        # 5. OPTIMISATION DE LA TAILLE
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        quality = 95

        # Premier essai à haute qualité
        output = BytesIO()

        if new_format == 'PNG':
            img.save(output, format="PNG", optimize=True)
        else:
            # Pour JPEG, gérer la conversion RGBA -> RGB si nécessaire
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                rgb_img.save(output, format="JPEG", quality=quality, optimize=True)
                corrections.append("Canal alpha supprimé (converti de RGBA à RGB)")
            else:
                img.save(output, format="JPEG", quality=quality, optimize=True)

        output.seek(0)
        data = output.getvalue()

        # Vérifier si l'image est trop grande et optimiser si nécessaire
        if len(data) > max_size_bytes:
            app.logger.error(f"Image is too large: {len(data) / 1024 / 1024:.2f}MB, optimizing...")

            # Pour JPEG, essayer de réduire la qualité d'abord
            if new_format == 'JPEG':
                for q in [85, 75, 65]:
                    output = BytesIO()

                    if img.mode == 'RGBA':
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])
                        rgb_img.save(output, format="JPEG", quality=q, optimize=True)
                    else:
                        img.save(output, format="JPEG", quality=q, optimize=True)

                    output.seek(0)
                    data = output.getvalue()

                    if len(data) <= max_size_bytes:
                        corrections.append(f"Qualité d'image réduite à {q}%")
                        break

            # Si toujours trop grande, réduire les dimensions
            if len(data) > max_size_bytes:
                scale = 0.9
                original_width, original_height = width, height

                while len(data) > max_size_bytes and scale > 0.3:  # Limite à 30% de la taille originale
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)

                    # S'assurer qu'on ne descend pas sous 300px
                    if min(new_width, new_height) < 300:
                        scale = 300 / min(original_width, original_height)
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                        break

                    # Redimensionner
                    resized = img.resize((new_width, new_height), Image.LANCZOS)

                    output = BytesIO()
                    if new_format == 'PNG':
                        resized.save(output, format="PNG", optimize=True)
                    else:
                        if resized.mode == 'RGBA':
                            rgb_img = Image.new('RGB', resized.size, (255, 255, 255))
                            rgb_img.paste(resized, mask=resized.split()[3])
                            rgb_img.save(output, format="JPEG", quality=quality, optimize=True)
                        else:
                            resized.save(output, format="JPEG", quality=quality, optimize=True)

                    output.seek(0)
                    data = output.getvalue()

                    if len(data) <= max_size_bytes:
                        corrections.append(
                            f"Image redimensionnée de {original_width}x{original_height} à {new_width}x{new_height}")
                        break

                    scale -= 0.1

            # Si toujours trop grande, forcer une forte compression
            if len(data) > max_size_bytes:
                if new_format == 'PNG':
                    # Convertir PNG en JPEG comme dernier recours
                    new_format = 'JPEG'
                    output = BytesIO()

                    if img.mode == 'RGBA':
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])
                        rgb_img.save(output, format="JPEG", quality=60, optimize=True)
                    else:
                        img.save(output, format="JPEG", quality=60, optimize=True)

                    output.seek(0)
                    data = output.getvalue()
                    corrections.append("Converti de PNG à JPEG pour réduire la taille")
                else:
                    # Déjà en JPEG, réduire encore plus la qualité
                    output = BytesIO()

                    if img.mode == 'RGBA':
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])
                        rgb_img.save(output, format="JPEG", quality=50, optimize=True)
                    else:
                        img.save(output, format="JPEG", quality=50, optimize=True)

                    output.seek(0)
                    data = output.getvalue()
                    corrections.append("Qualité d'image fortement réduite pour respecter la limite de taille")

        # Si toujours trop grande après toutes les optimisations
        if len(data) > max_size_bytes:
            return {
                "success": False,
                "error_code": "IMAGE_TOO_LARGE",
                "error_message": f"Impossible d'optimiser l'image en dessous de {max_size_mb}MB. L'image est actuellement de {len(data) / 1024 / 1024:.2f}MB après optimisation."
            }

        # Encoder en base64
        encoded = base64.b64encode(data).decode('utf-8')
        encoded_size_mb = len(encoded) / 1024 / 1024

        # Log final
        app.logger.error(
            f"Final image: format={new_format}, size={len(data) / 1024 / 1024:.2f}MB, encoded={encoded_size_mb:.2f}MB")

        # Préfixe MIME
        mime_type = "image/jpeg" if new_format == 'JPEG' else "image/png"
        base64_image = f"data:{mime_type};base64,{encoded}"

        # Vérifications finales
        if encoded_size_mb > 20:  # Limite absolue de MiniMax
            return {
                "success": False,
                "error_code": "ENCODED_SIZE_LIMIT",
                "error_message": f"L'image encodée dépasse la limite de 20MB imposée par le service. Taille actuelle: {encoded_size_mb:.2f}MB."
            }

        # Succès
        return {
            "success": True,
            "image": base64_image,
            "format": new_format,
            "dimensions": f"{img.size[0]}x{img.size[1]}",
            "ratio": round(img.size[0] / img.size[1], 2),
            "size_mb": round(len(data) / 1024 / 1024, 2),
            "encoded_size_mb": round(encoded_size_mb, 2),
            "corrections": corrections if corrections else ["Aucune correction nécessaire"]
        }

    except Exception as e:
        import traceback
        app.logger.error(f"Error processing image: {str(e)}")
        app.logger.error(traceback.format_exc())

        return {
            "success": False,
            "error_code": "PROCESSING_ERROR",
            "error_message": f"Erreur lors du traitement de l'image: {str(e)}"
        }

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

        """TEMP"""
        headers = {
            'authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }


        """/TEMP"""


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
                    base64_image = f"data:image/jpeg;base64,{base64_image}"

                # Vérifier la taille de l'image encodée
                image_size_mb = len(base64_image) / 1024 / 1024
                app.logger.error(f"Base64 image size: {image_size_mb:.2f} MB")

                payload["first_frame_image"] = base64_image

            # Nettoyage et validation des autres paramètres
            supported_params = ["prompt_optimizer", "callback_url"]
            for param in supported_params:
                if param in additional_params:
                    # Conversion en booléen pour les paramètres qui le nécessitent
                    if param in ["prompt_optimizer",]:
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
        app.logger.error(f"Payload parameters: {str(payload)[:200]}")
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
                    # Vérifier quelle partie du èpad pose problème
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