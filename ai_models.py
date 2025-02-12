from abc import ABC, abstractmethod
import openai
import json
import requests
from flask import current_app as app
import os
from datetime import datetime
import uuid
import redis

# Configuration Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False
)


class AIModelGenerator(ABC):
    @abstractmethod
    def generate(self, prompt, additional_params=None):
        pass


class DallEGenerator(AIModelGenerator):
    def __init__(self, api_key, model="dall-e-3"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def generate(self, prompt, additional_params=None):
        try:
            params = {
                "model": self.model,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            }
            if additional_params:
                params.update(additional_params)

            response = openai.images.generate(**params)
            return {
                "success": True,
                "image_url": response.data[0].url,
                "model": "dall-e"
            }
        except Exception as e:
            app.logger.error(f"DALL-E generation error: {str(e)}")
            return {"success": False, "error": str(e)}


class MidjourneyGenerator(AIModelGenerator):
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.redis = redis_client

    def generate(self, prompt, additional_params=None):
        try:
            task_id = str(uuid.uuid4())  # Identifiant unique pour suivre la génération

            # Envoi au webhook Make
            payload = {
                 "prompt": prompt
            }

            # Log pour debug
            app.logger.error(f"Sending to Make/Userapi.AI: {payload}")

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                app.logger.info("Successfully sent to Make webhook")
                metadata = {
                    'type': b'generated',
                    'prompt': prompt.encode('utf-8'),
                    'timestamp': datetime.now().isoformat().encode('utf-8'),
                    'model': 'midjourney'.encode('utf-8'),
                    'status': 'processing'.encode('utf-8'),
                    'task_id': task_id.encode('utf-8')
                }

                # Stockage du statut dans Redis
                self.redis.hmset(
                    f"midjourney_task:{task_id}",
                    metadata
                )
                self.redis.expire(f"midjourney_task:{task_id}", 3600)  # expire après 1h

                return {
                    "success": True,
                    "status": "processing",
                    "task_id": task_id
                }
            else:
                raise ValueError(f"Discord webhook error: {response.status_code}")

        except Exception as e:
            app.logger.error(f"Midjourney generation error: {str(e)}")
            return {"success": False, "error": str(e)}

class AIModelManager:
    def __init__(self):
        self.generators = {}

    def register_model(self, model_name, generator):
        """Enregistre un nouveau générateur de modèle"""
        self.generators[model_name] = generator

    def generate_image(self, model_name, prompt, additional_params=None):
        """Génère une image avec le modèle spécifié"""
        if model_name not in self.generators:
            raise ValueError(f"Model {model_name} not found")

        generator = self.generators[model_name]
        return generator.generate(prompt, additional_params)


# Fonction d'initialisation pour créer et configurer le gestionnaire
def create_ai_manager():
    manager = AIModelManager()

    # Configuration DALL-E
    dalle_api_key = os.getenv("OPENAI_API_KEY")
    if dalle_api_key:
        manager.register_model("dall-e", DallEGenerator(dalle_api_key))

    # Configuration Midjourney
    midjourney_webhook = os.getenv("MAKE_WEBHOOK_URL")
    if midjourney_webhook:
        manager.register_model("midjourney", MidjourneyGenerator(midjourney_webhook))

    return manager

