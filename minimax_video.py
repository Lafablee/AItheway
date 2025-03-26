import requests
import json
import time
import os
import logging
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
            task_id = self.create_generation_task(prompt, model, additional_params)
            if task_id:
                return {"success": True, "task_id": task_id, "status": "processing"}
            return {"success": False, "error": "Failed to create generation task"}
        except Exception as e:
            app.logger.error(f"Error in MiniMaxVideoGenerator.generate_async: {str(e)}")
            return {"success": False, "error": str(e)}

    def create_generation_task(self, prompt, model=None, additional_params=None):
        """Crée une tâche de génération et retourne le task_id"""
        headers = {
            'authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
        }

        # Ajouter les paramètres additionnels
        if additional_params:
            if isinstance(additional_params, str):
                try:
                    additional_params = json.loads(additional_params)
                except json.JSONDecodeError:
                    app.logger.error(f"Failed to parse additional_params JSON: {additional_params}")
                    additional_params = {}

            # Gestion de l'image de référence
            if "first_frame_image" in additional_params:
                # Transférer directement l'image encodée en base64
                payload["first_frame_image"] = additional_params["first_frame_image"]
                app.logger.error(
                    f"Adding base64 image to payload, length: {len(additional_params['first_frame_image'])}")

            # Gérer le paramètre subject_reference (doit être un booléen)
            if "subject_reference" in additional_params:
                if isinstance(additional_params["subject_reference"], str):
                    payload["subject_reference"] = additional_params["subject_reference"].lower() == 'true'
                else:
                    payload["subject_reference"] = bool(additional_params["subject_reference"])
                app.logger.error(f"Adding subject_reference to payload: {payload['subject_reference']}")

            # Ajouter les autres paramètres supportés par l'API
            supported_params = ["prompt_optimizer", "callback_url"]
            for param in supported_params:
                if param in additional_params:
                    if param == "prompt_optimizer" and isinstance(additional_params[param], str):
                        # Convertir les chaînes 'true'/'false' en booléens
                        payload[param] = additional_params[param].lower() == 'true'
                    else:
                        payload[param] = additional_params[param]
                    app.logger.error(f"Adding parameter {param} to payload: {payload[param]}")

        app.logger.info(f"Creating MiniMax video generation task with payload keys: {list(payload.keys())}")

        try:
            response = requests.post(
                f"{self.base_url}/video_generation",
                headers=headers,
                json=payload  # Utiliser json au lieu de data pour gérer automatiquement la sérialisation
            )

            response_data = response.json()
            app.logger.info(f"MiniMax API response: {response_data}")

            if response.status_code == 200 and response_data.get("task_id"):
                return response_data.get("task_id")
            else:
                app.logger.error(f"Error creating MiniMax task: {response_data}")
                return None

        except Exception as e:
            app.logger.error(f"Exception in create_generation_task: {str(e)}")
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