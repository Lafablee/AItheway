# tasks.py
import asyncio
import time
import json
import re
from datetime import datetime, timedelta
import aiohttp
import os
import redis


from config import redis_client, TEMP_STORAGE_DURATION
from ai_models import MidjourneyGenerator

app = None

async def check_pending_midjourney_tasks():
    """
        Background task to check for pending Midjourney tasks and attempt to complete them
        by fetching their upscaled images if they're ready.
        """
    global app

    while True:
        try:
            # Find all pending Midjourney tasks
            pending_tasks = []
            for key in redis_client.keys("midjourney_group:*"):
                try:
                    group_data = json.loads(redis_client.get(key))
                    task_id = group_data.get('task_id')
                    status = group_data.get('status')

                    if status == 'pending' and len(group_data.get('images', [])) < 4:
                        # Check if task is not too old (within last 2 hours)
                        timestamp = datetime.fromisoformat(group_data.get('timestamp'))
                        if datetime.now() - timestamp < timedelta(hours=2):
                            pending_tasks.append(task_id)
                except:
                    continue

            app.logger.error(f"Found {len(pending_tasks)} pending Midjourney tasks")

            # Process each pending task
            for task_id in pending_tasks:
                app.logger.error(f"Checking pending task: {task_id}")

                # Get task metadata
                metadata_key = f"midjourney_task:{task_id}"
                metadata = redis_client.hgetall(metadata_key)

                if not metadata:
                    continue

                # Check if upscales were triggered
                upscale_triggered = False
                for key in metadata:
                    if key.startswith(b'upscale_status_') and metadata[key] == b'triggered':
                        upscale_triggered = True
                        break

                if not upscale_triggered:
                    app.logger.error(f"Task {task_id} has no triggered upscales, skipping")
                    continue

                # Attempt to fetch upscaled images
                # Create a Midjourney generator instance
                discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
                discord_auth_token = os.getenv("DISCORD_AUTH_TOKEN")
                discord_guild_id = os.getenv("DISCORD_GUILD_ID")

                if not all([discord_channel_id, discord_auth_token, discord_guild_id]):
                    app.logger.error("Missing Discord credentials for background task")
                    continue

                midjourney_generator = MidjourneyGenerator(
                    channel_id=discord_channel_id,
                    auth_token=discord_auth_token,
                    guild_id=discord_guild_id
                )

                # Create a session and call poll_for_upscaled_images
                async with aiohttp.ClientSession(headers={
                    "Authorization": discord_auth_token,
                    "Content-Type": "application/json"
                }) as session:
                    try:
                        prompt = metadata.get(b'prompt', b'').decode('utf-8')
                        app.logger.error(f"Polling for upscaled images for task {task_id}")

                        await midjourney_generator.poll_for_upscaled_images(
                            session=session,
                            message_id=None,  # We don't need the original message ID for polling
                            task_id=task_id,
                            max_attempts=3,  # Just a few attempts per cycle
                            delay=5  # Short delay between attempts
                        )
                    except Exception as e:
                        app.logger.error(f"Error polling for task {task_id}: {str(e)}")

        except Exception as e:
            app.logger.error(f"Error in background task: {str(e)}")

        # Sleep for a while before next check
        await asyncio.sleep(300)  # Check every 5 minutes


async def check_pending_video_tasks():
    """Vérifie périodiquement les tâches de génération vidéo en attente."""
    # Attendre un peu pour s'assurer que l'application est complètement démarrée
    await asyncio.sleep(10)

    while True:
        try:
            app.logger.info("Checking pending video generation tasks")

            # Récupérer toutes les tâches vidéo en cours
            from video_manager import VideoManager
            from app import storage_manager, ai_manager

            # Initialiser les gestionnaires
            video_manager = VideoManager(storage_manager)
            minimax_video_generator = ai_manager.generators.get(
                "minimax-video").generator if "minimax-video" in ai_manager.generators else None

            if not minimax_video_generator:
                app.logger.warning("MiniMax video generator not available")
                await asyncio.sleep(60)
                continue

            # Récupérer les tâches en cours via Redis
            # On cherche les clés des vidéos avec status="processing"
            redis_client = storage_manager.redis
            processing_keys = []

            # Méthode 1: Utiliser les métadonnées
            all_video_keys = redis_client.keys("video:temp:*")
            for key_bytes in all_video_keys:
                try:
                    key = key_bytes.decode('utf-8') if isinstance(key_bytes, bytes) else key_bytes
                    metadata = storage_manager.get_metadata(key)
                    if metadata and metadata.get('status') == 'processing':
                        processing_keys.append((key, metadata))
                except Exception as e:
                    app.logger.error(f"Error checking video metadata for {key_bytes}: {e}")

            app.logger.info(f"Found {len(processing_keys)} pending video tasks")

            # Pour chaque tâche en cours
            for key, metadata in processing_keys:
                try:
                    # Récupérer le task_id
                    task_id = metadata.get('task_id')
                    if not task_id:
                        app.logger.warning(f"No task_id in metadata for {key}")
                        continue

                    # Vérifier l'état auprès de l'API MiniMax
                    app.logger.info(f"Checking status of video task {task_id}")
                    status_response = minimax_video_generator.check_generation_status(task_id)

                    if status_response.get('success'):
                        status = status_response.get('status', '')
                        file_id = status_response.get('file_id', '')

                        # Si la tâche est terminée
                        if status == "Success" and file_id:
                            # Récupérer l'URL de téléchargement
                            download_url = minimax_video_generator.get_download_url(file_id)

                            # Mettre à jour les métadonnées
                            video_manager.update_video_status(
                                task_id,
                                'completed',
                                file_id,
                                download_url
                            )

                            app.logger.info(f"Video task {task_id} completed, download URL: {download_url}")

                            # Option: télécharger automatiquement la vidéo
                            # Cette partie est optionnelle car on peut aussi télécharger à la demande
                            if download_url and metadata.get('download_on_complete', False):
                                try:
                                    video_data = video_manager.download_from_url(download_url)
                                    if video_data:
                                        video_manager.store_video_file(key, video_data)
                                        app.logger.info(f"Auto-downloaded video for {task_id}")
                                except Exception as e:
                                    app.logger.error(f"Error auto-downloading video: {e}")

                        elif status == "Fail":
                            # Mettre à jour le statut en cas d'échec
                            video_manager.update_video_status(task_id, 'failed')
                            app.logger.warning(f"Video task {task_id} failed")

                        else:
                            # Mettre à jour le statut
                            video_manager.update_video_status(task_id, status)
                            app.logger.info(f"Video task {task_id} status: {status}")

                except Exception as e:
                    app.logger.error(f"Error processing video task {key}: {e}")

            # Attendre avant la prochaine vérification
            await asyncio.sleep(60)  # Vérifier toutes les minutes

        except Exception as e:
            app.logger.error(f"Error in check_pending_video_tasks: {e}")
            await asyncio.sleep(30)

def start_background_tasks(app):
    loop = asyncio.get_event_loop()
    loop.create_task(check_pending_midjourney_tasks(app))
    loop.create_task(check_pending_video_tasks(app))
