from abc import ABC, abstractmethod
import openai
import json
import requests
from flask import current_app as app
import os
from datetime import datetime
import asyncio
import aiohttp
import redis.asyncio as aioredis
from redis.asyncio import Redis
import time
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

    # Méthode synchrone pour la compatibilité avec les modèles existants
    def generate_sync(self, prompt, additional_params=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate(prompt, additional_params))
        finally:
            loop.close()

class DallEGenerator(AIModelGenerator):
    def __init__(self, api_key, model="dall-e-3"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    async def generate(self, prompt, additional_params=None):
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
    def __init__(self, channel_id, auth_token, guild_id):
        self.channel_id = channel_id
        self.auth_token = auth_token
        self.guild_id = guild_id
        self.redis = redis_client
        self.MIDJOURNEY_BOT_ID = "936929561302675456"
        self.async_redis = None

    async def get_async_redis(self) -> Redis:
        """Lazy initialization of async Redis connection"""
        if self.async_redis is None:
            self.async_redis = await aioredis.from_url(
                'redis://localhost'
            )
        return self.async_redis

    async def get_application_commands(self, session):
        """Récupère les commandes disponibles et leur version actuelle"""
        url = f"https://discord.com/api/v9/applications/{self.MIDJOURNEY_BOT_ID}/commands"
        async with session.get(url) as response:
            if response.status == 200:
                commands = await response.json()
                for command in commands:
                    if command.get('name') == 'imagine':
                        return command.get('id'), command.get('version')
        return None, None

    async def wait_for_midjourney_response(self, session, prompt, start_time):
        """Attend et récupère la réponse de Midjourney avec les 4 images"""
        for attempt in range(20):
            await asyncio.sleep(6)

            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=30)
            async with session.get(
                    f"https://discord.com/api/v9/channels/{self.channel_id}/messages?limit=10",
                    timeout=timeout
            ) as msg_response:
                if msg_response.status == 200:
                    messages = await msg_response.json()

                    for message in messages:
                        author = message.get('author', {})
                        content = message.get('content', '')
                        message_time = int(
                            datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00')).timestamp()
                        ) if message.get('timestamp') else 0

                        if (message_time < start_time):
                            continue

                        if (author.get('id') == self.MIDJOURNEY_BOT_ID and
                                message.get('components') and
                                message.get('attachments') and
                                prompt.lower() in content.lower()):
                            return message

        return None

    async def upscale_image(self, session, message_id, custom_id):
        """Upscale une image spécifique"""
        payload = {
            "type": 3,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "message_id": message_id,
            "application_id": self.MIDJOURNEY_BOT_ID,
            "session_id": "1234567890",
            "data": {
                "component_type": 2,
                "custom_id": custom_id
            }
        }

        async with session.post(
                "https://discord.com/api/v9/interactions",
                json=payload
        ) as response:
            return response.status == 204

    async def handle_failed_upscale(self, task_id, variation_number):
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                await self.upscale_image(...)
                return True
            except Exception as e:
                retry_count += 1
                await asyncio.sleep(5 * retry_count)  # Backoff exponentiel

        return False

    async def generate(self, prompt, additional_params=None):
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            metadata_key = f"midjourney_task:{task_id}"
            redis = await self.get_async_redis()

            # Store initial metadata
            metadata = {
                'type': b'generated',
                'prompt': prompt.encode('utf-8'),
                'timestamp': datetime.now().isoformat().encode('utf-8'),
                'model': b'midjourney',
                'status': b'processing',
                'task_id': task_id.encode('utf-8'),
                'parameters': json.dumps({
                    'model': 'midjourney',
                    'size': '1024x1024'
                }).encode('utf-8')
            }

            # Utilisez un pipeline Redis asynchrone
            tr = redis.multi_exec()
            await tr.hmset(metadata_key, metadata)
            await tr.expire(metadata_key, 3600)
            await tr.execute()

            # Setup Discord session
            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                # Get command version
                command_id, command_version = await self.get_application_commands(session)
                if not command_id or not command_version:
                    await redis.delete(metadata_key)
                    raise ValueError("Impossible de récupérer la version de la commande Midjourney")

                # Prepare /imagine payload
                imagine_payload = {
                    "type": 2,
                    "application_id": self.MIDJOURNEY_BOT_ID,
                    "guild_id": self.guild_id,
                    "channel_id": self.channel_id,
                    "session_id": "1234567890",
                    "data": {
                        "version": command_version,
                        "id": command_id,
                        "name": "imagine",
                        "type": 1,
                        "options": [
                            {
                                "type": 3,
                                "name": "prompt",
                                "value": prompt
                            }
                        ]
                    }
                }

                start_time = int(time.time())

                # Send /imagine command
                async with session.post(
                        "https://discord.com/api/v9/interactions",
                        json=imagine_payload
                ) as response:
                    if response.status == 204:
                        # Wait for initial message with 4 images
                        message = await self.wait_for_midjourney_response(session, prompt, start_time)

                        if message:
                            message_id = message.get('id')
                            components = message.get('components', [])
                            initial_image_url = message.get('attachments', [{}])[0].get('url')

                            # Create image group
                            group_data = {
                                'task_id': task_id,
                                'prompt': prompt,
                                'initial_grid': initial_image_url,
                                'images': [],
                                'timestamp': datetime.now().isoformat(),
                                'status': 'pending'
                            }

                            # Store group data
                            await redis.setex(
                                f"midjourney_group:{task_id}",
                                3600,
                                json.dumps(group_data)
                            )

                            # Extract upscale buttons
                            upscale_buttons = []
                            for component in components:
                                for button in component.get('components', []):
                                    if button.get('custom_id', '').startswith('MJ::JOB::upsample::'):
                                        upscale_buttons.append(button.get('custom_id'))

                            if upscale_buttons:
                                # Trigger upscales
                                for i, button_id in enumerate(upscale_buttons, 1):
                                    if await self.upscale_image(session, message_id, button_id):
                                        # Update status in Redis
                                        await redis.hset(metadata_key, 'upscale_status', f'U{i}_processing')

                            return {
                                "success": True,
                                "status": "processing",
                                "task_id": task_id
                            }

                    return {
                        "success": False,
                        "error": "Failed to send command to Discord"
                    }

        except Exception as e:
            # Clean up Redis data in case of error
            if metadata_key:
                redis = await self.get_async_redis()
                await redis.delete(metadata_key)
            return {"success": False, "error": str(e)}
        finally:
            # Fermer la connexion Redis asynchrone si elle existe
            if self.async_redis is not None:
                self.async_redis.close()
                await self.async_redis.wait_closed()
                self.async_redis = None

class AIModelManager:
    def __init__(self):
        self.generators = {}

    def register_model(self, model_name, generator):
        """Enregistre un nouveau générateur de modèle"""
        self.generators[model_name] = generator

    async def generate_image(self, model_name, prompt, additional_params=None):
        """Génère une image avec le modèle spécifié"""
        if model_name not in self.generators:
            raise ValueError(f"Model {model_name} not found")

        generator = self.generators[model_name]
        return await generator.generate(prompt, additional_params)

    def generate_image_sync(self, model_name, prompt, additional_params=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_image(model_name, prompt, additional_params))
        finally:
            loop.close()

# Fonction d'initialisation pour créer et configurer le gestionnaire
def create_ai_manager():
    manager = AIModelManager()

    # Configuration DALL-E
    dalle_api_key = os.getenv("openai.api_key")
    print(f"DALL-E API Key present: {bool(dalle_api_key)}")
    if dalle_api_key:
        manager.register_model("dall-e", DallEGenerator(dalle_api_key))

    # Configuration Midjourney
    discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
    discord_auth_token = os.getenv("DISCORD_AUTH_TOKEN")
    discord_guild_id = os.getenv("DISCORD_GUILD_ID")

    if all([discord_channel_id, discord_auth_token, discord_guild_id]):
        manager.register_model(
            "midjourney",
            MidjourneyGenerator(
                channel_id=discord_channel_id,
                auth_token=discord_auth_token,
                guild_id=discord_guild_id
            )
        )
    return manager

