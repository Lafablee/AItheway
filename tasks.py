# tasks.py
import asyncio
import json
import re
from datetime import datetime, timedelta
import aiohttp
import os

from app import app, redis_client
from ai_models import MidjourneyGenerator

async def check_pending_midjourney_tasks():
    """
        Background task to check for pending Midjourney tasks and attempt to complete them
        by fetching their upscaled images if they're ready.
        """
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

def start_background_tasks(app):
    loop = asyncio.get_event_loop()
    loop.create_task(check_pending_midjourney_tasks(app))
