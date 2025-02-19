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
            internal_task_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # Créer la structure initiale du groupe
            group_data = {
                'internal_task_id': internal_task_id,
                'prompt': prompt,
                'status': 'processing',
                'timestamp': timestamp,
                'images': [],
                'total_expected': 4
            }

            # Stocker le groupe
            self.redis.setex(
                f"midjourney_group:{internal_task_id}",
                3600,
                json.dumps(group_data)
            )

            # Stocker la tâche active avec son prompt
            active_task_data = {
                'prompt': prompt,
                'internal_task_id': internal_task_id,
                'timestamp': timestamp,
                'status': 'processing'
            }

            # On stocke cette tâche comme la tâche active la plus récente
            self.redis.setex(
                'midjourney_active_task',
                3600,
                json.dumps(active_task_data)
            )

            # Envoi à Make/userapi.ai
            response = requests.post(
                self.webhook_url,
                json={"prompt": prompt},
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code != 200:
                raise ValueError(f"Webhook error: {response.status_code}")

            return {
                "success": True,
                "status": "processing",
                "task_id": internal_task_id
            }

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

