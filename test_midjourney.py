import os
import asyncio
import json
import aiohttp
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv(dotenv_path='.env')

# Récupération des variables d'environnement
channel_id = os.getenv('DISCORD_CHANNEL_ID')
auth_token = os.getenv('DISCORD_AUTH_TOKEN')
guild_id = os.getenv('DISCORD_GUILD_ID')
MIDJOURNEY_BOT_ID = "936929561302675456"

# Validation des variables d'environnement
if not channel_id or not auth_token or not guild_id:
    raise ValueError(
        f"Variables d'environnement manquantes:\n"
        f"DISCORD_CHANNEL_ID: {'✓' if channel_id else '✗'}\n"
        f"DISCORD_AUTH_TOKEN: {'✓' if auth_token else '✗'}\n"
        f"DISCORD_GUILD_ID: {'✓' if guild_id else '✗'}"
    )


async def get_application_commands(session):
    """Récupère les commandes disponibles et leur version actuelle"""
    url = f"https://discord.com/api/v9/applications/{MIDJOURNEY_BOT_ID}/commands"
    async with session.get(url) as response:
        if response.status == 200:
            commands = await response.json()
            for command in commands:
                if command.get('name') == 'imagine':
                    return command.get('id'), command.get('version')
    return None, None


async def wait_for_midjourney_response(session, channel_id, prompt, start_time):
    """Attend et récupère la réponse de Midjourney avec les 4 images"""
    for attempt in range(20):  # 20 tentatives (environ 2 minutes)
        await asyncio.sleep(6)  # Augmentation du délai entre les tentatives
        logger.info(f"Tentative {attempt + 1}/20 de récupération du message...")

        async with session.get(
                f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=10"
        ) as msg_response:
            if msg_response.status == 200:
                messages = await msg_response.json()
                if not isinstance(messages, list):
                    logger.error(f"Format de réponse inattendu: {messages}")
                    continue

                for message in messages:
                    author = message.get('author', {})
                    content = message.get('content', '')
                    # Conversion du timestamp ISO 8601 en timestamp Unix
                    if message.get('timestamp'):
                        message_time = int(
                            datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00')).timestamp())
                    else:
                        message_time = 0

                    # Vérifie si le message est plus récent que notre commande
                    if message_time < start_time:
                        continue

                    if (author.get('id') == MIDJOURNEY_BOT_ID and
                            message.get('components') and
                            message.get('attachments') and
                            prompt.lower() in content.lower()):  # Vérifie si le prompt correspond
                        logger.info("Message Midjourney correspondant trouvé avec composants et images")
                        return message

                logger.info("Aucun message correspondant trouvé dans cette tentative")
            else:
                logger.error(f"Erreur lors de la récupération des messages: {msg_response.status}")

    logger.error("Timeout: Aucun message Midjourney trouvé après 20 tentatives")
    return None


async def wait_for_upscaled_image(session, channel_id, start_time):
    """Attend l'image upscalée"""
    for attempt in range(15):  # 15 tentatives
        await asyncio.sleep(4)
        logger.info(f"Tentative {attempt + 1}/15 de récupération de l'image upscalée...")

        async with session.get(
                f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=5"
        ) as msg_response:
            if msg_response.status == 200:
                messages = await msg_response.json()

                for message in messages:
                    # Conversion du timestamp ISO 8601 en timestamp Unix
                    if message.get('timestamp'):
                        message_time = int(
                            datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00')).timestamp())
                    else:
                        message_time = 0

                    # Vérifie si le message est plus récent que notre commande d'upscale
                    if (message_time > start_time and
                            message.get('author', {}).get('id') == MIDJOURNEY_BOT_ID and
                            message.get('attachments')):
                        return message

    return None


async def upscale_image(session, channel_id, message_id, custom_id):
    """Upscale une image spécifique"""
    payload = {
        "type": 3,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "message_id": message_id,
        "application_id": MIDJOURNEY_BOT_ID,
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


async def test_midjourney_communication():
    logger.info("Démarrage du test avec les configurations:")
    logger.info(f"Channel ID: {channel_id}")
    logger.info(f"Guild ID: {guild_id}")
    logger.info(f"Auth Token (premiers caractères): {auth_token[:10]}...")

    headers = {
        "Authorization": auth_token,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    prompt = "a cute cat playing with a red ball, high quality, 4k --v 5"

    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=False)) as session:
        # Test d'authentification
        logger.info("Test de l'authentification Discord...")
        async with session.get("https://discord.com/api/v9/users/@me") as response:
            if response.status == 401:
                logger.error("Erreur d'authentification. Vérifiez votre token Discord")
                return
            elif response.status == 200:
                user_data = await response.json()
                logger.info(f"Authentification réussie! Connecté en tant que: {user_data.get('username')}")
            else:
                logger.error(f"Statut inattendu: {response.status}")
                return

        # Récupération de la version actuelle de la commande
        command_id, command_version = await get_application_commands(session)
        if not command_id or not command_version:
            logger.error("Impossible de récupérer la version de la commande Midjourney")
            return

        logger.info(f"Version de la commande récupérée: {command_version}")

        # Commande /imagine
        imagine_payload = {
            "type": 2,
            "application_id": MIDJOURNEY_BOT_ID,
            "guild_id": guild_id,
            "channel_id": channel_id,
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

        logger.info("Envoi de la commande /imagine...")
        start_time = int(time.time())

        async with session.post(
                "https://discord.com/api/v9/interactions",
                json=imagine_payload
        ) as response:
            status = response.status
            logger.info(f"Statut de la réponse /imagine: {status}")

            if status == 204:
                logger.info("Commande /imagine envoyée avec succès! Attente de la génération...")

                # Boucle principale d'attente pour la génération d'image
                message = None
                generation_attempts = 0
                max_generation_attempts = 30  # 30 * 6 secondes = 3 minutes d'attente max

                while not message and generation_attempts < max_generation_attempts:
                    generation_attempts += 1
                    logger.info(f"Tentative de génération {generation_attempts}/{max_generation_attempts}")
                    message = await wait_for_midjourney_response(session, channel_id, prompt, start_time)
                    if not message:
                        await asyncio.sleep(6)  # Attendre avant la prochaine tentative

                if message:
                    logger.info("Message de Midjourney avec images trouvé!")
                    message_id = message.get('id')

                    # Extraire les custom_id des boutons U1-U4
                    components = message.get('components', [])
                    upscale_buttons = []
                    for component in components:
                        for button in component.get('components', []):
                            if button.get('custom_id', '').startswith('MJ::JOB::upsample::'):
                                upscale_buttons.append(button.get('custom_id'))

                    if upscale_buttons:
                        # Upscale de toutes les images (U1 à U4)
                        upscaled_urls = []
                        for i, button_id in enumerate(upscale_buttons, 1):
                            logger.info(f"Tentative d'upscale de U{i}...")
                            upscale_start_time = int(time.time())

                            if await upscale_image(session, channel_id, message_id, button_id):
                                logger.info(f"Commande d'upscale U{i} envoyée avec succès!")

                                # Attendre et récupérer l'image upscalée avec plusieurs tentatives
                                logger.info(f"Attente de l'image U{i} upscalée...")
                                upscaled_message = None
                                upscale_attempts = 0
                                max_upscale_attempts = 20  # 20 * 4 secondes = 80 secondes d'attente max

                                while not upscaled_message and upscale_attempts < max_upscale_attempts:
                                    upscale_attempts += 1
                                    logger.info(f"Tentative d'upscale {upscale_attempts}/{max_upscale_attempts}")
                                    upscaled_message = await wait_for_upscaled_image(session, channel_id,
                                                                                     upscale_start_time)
                                    if not upscaled_message:
                                        await asyncio.sleep(4)

                                if upscaled_message and upscaled_message.get('attachments'):
                                    logger.info(f"Image U{i} upscalée trouvée!")
                                    url = upscaled_message['attachments'][0].get('url')
                                    logger.info(f"URL de l'image U{i} upscalée: {url}")
                                    upscaled_urls.append(url)
                                else:
                                    logger.error(
                                        f"Impossible de trouver l'image U{i} upscalée après toutes les tentatives")
                            else:
                                logger.error(f"Échec de l'upscale U{i}")

                            # Attendre un peu entre chaque upscale pour éviter les rate limits
                            await asyncio.sleep(2)

                        # Afficher le récapitulatif de toutes les URLs
                        if upscaled_urls:
                            logger.info("\nRécapitulatif des images upscalées:")
                            for i, url in enumerate(upscaled_urls, 1):
                                logger.info(f"U{i}: {url}")
                    else:
                        logger.error("Aucun bouton d'upscale trouvé dans le message")
                else:
                    logger.error("Pas de réponse de Midjourney après toutes les tentatives de génération")
            else:
                response_text = await response.text()
                logger.error(f"Erreur lors de l'envoi de la commande /imagine: {response_text}")


if __name__ == "__main__":
    asyncio.run(test_midjourney_communication())