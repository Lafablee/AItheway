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
import re

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
                'redis://localhost',
                encoding='utf-8',
                decode_responses=True
            )
        return self.async_redis

    async def __aenter__(self):
        self.redis = await self.get_async_redis()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis:
            await self.redis.close()

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
        # Calculate initial delay based on prompt complexity
        initial_delay = self.calculate_prompt_complexity(prompt)

        # Add initial waiting period before looking for the response
        app.logger.error(f"Waiting {initial_delay} seconds before checking for Midjourney response...")
        await asyncio.sleep(initial_delay)
        for attempt in range(20):
            await asyncio.sleep(6)

            try:
                timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=60)
                async with session.get(
                        f"https://discord.com/api/v9/channels/{self.channel_id}/messages?limit=10",
                        timeout=timeout
                ) as msg_response:
                    if msg_response.status == 200:
                        app.logger.error("code 200 from miidjourney generation")

                        messages = await msg_response.json()
                        app.logger.error(f"Got {len(messages)} messages")

                        for message in messages:
                            author = message.get('author', {})
                            content = message.get('content', '')
                            message_time = int(
                                datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00')).timestamp()
                            ) if message.get('timestamp') else 0

                            if (author.get('id') == self.MIDJOURNEY_BOT_ID and
                                    message.get('components') and
                                    message.get('attachments') and
                                    prompt.lower() in content.lower()):
                                app.logger.error(f"Found Midjourney response with components: {message['components']}")
                                return message
                        app.logger.error("No matching message found in this batch")
            except Exception as e:
                app.logger.error(f"Error in wait_for_midjourney_response: {str(e)}")
            app.logger.error(f"Attempt {attempt + 1} complete, trying again...")
        return None

    def calculate_prompt_complexity(self, prompt):
        """Calculates waiting time based on prompt complexity"""
        # Base delay for all prompts
        base_delay = 15  # 15 seconds minimum waiting time

        # Initialize complexity factors
        word_count = len(prompt.split())
        complexity_score = 0

        # 1. Word count factor
        if word_count > 50:
            complexity_score += 3
        elif word_count > 30:
            complexity_score += 2
        elif word_count > 15:
            complexity_score += 1

        # 2. Check for complexity indicators
        complexity_indicators = [
            'realistic', 'detailed', 'intricate', 'high resolution', '8k',
            'landscape', 'cityscape', 'panorama',
            'group', 'crowd', 'multiple',
            'reflection', 'glass', 'water', 'transparent',
            'cinematic', 'dramatic lighting', 'sunset', 'night scene',
            'fog', 'rain', 'snow', 'particles'
        ]

        indicator_count = sum(1 for indicator in complexity_indicators
                              if indicator.lower() in prompt.lower())

        # Add score based on indicators
        if indicator_count > 7:
            complexity_score += 3
        elif indicator_count > 4:
            complexity_score += 2
        elif indicator_count > 2:
            complexity_score += 1

        # Calculate additional delay (max 15 seconds additional)
        additional_delay = min(complexity_score * 3, 15)
        total_delay = base_delay + additional_delay

        app.logger.error(f"Prompt complexity: {complexity_score}, Total delay: {total_delay}s")
        return total_delay

    async def upscale_image(self, session, message_id, custom_id):
        """Upscale une image spécifique"""
        #timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)


        try:
            app.logger.error(f"Atttempting to upscale image with message_id= {message_id}, and custom_id={custom_id}")
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
            app.logger.error(f"Sending upscale request with payload: {json.dumps(payload)}")

            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=60)
            async with session.post(
                    "https://discord.com/api/v9/interactions",
                    json=payload,
                    timeout=timeout
            ) as response:
                success = response.status == 204
                app.logger.error(f"Upscale request result: status={response.status}, success={success}")

                if not success:
                    error_text = await response.text()
                    app.logger.error(f"Error response: {error_text}")
                return success
        except Exception as e:
            app.logger.error(f"Error in upscale_image: {str(e)}")
            return False

    async def handle_upscales(self, session, message_id, upscale_buttons, task_id):
        """Handle all upscale operations with retry logic"""
        results = []

        for i, button_id in enumerate(upscale_buttons, 1):
            retry_count = 0
            max_retries = 3
            success = False

            while retry_count < max_retries and not success:
                try:
                    app.logger.error(f"Attempting upscale U{i} with button ID: {button_id}")
                    success = await self.upscale_image(session, message_id, button_id)
                    if success:
                        app.logger.error(f"Successfully triggered upscale U{i}")
                        # Update status in Redis to track progress
                        results.append({"button": f"U{i}", "success": True})
                        # Add a small delay between upscale requests to avoid rate limiting
                        await asyncio.sleep(1)
                    else:
                        app.logger.error(f"Failed to trigger upscale U{i}, retrying...")
                        retry_count += 1
                        await asyncio.sleep(2 * retry_count)  # Exponential backoff
                except Exception as e:
                    app.logger.error(f"Error during upscale U{i}: {str(e)}")
                    retry_count += 1
                    await asyncio.sleep(2 * retry_count)

            if not success:
                results.append({"button": f"U{i}", "success": False})

        return results

    async def poll_for_upscaled_images(self, session, message_id, task_id, max_attempts=30, delay=10):
        """
        Enhanced polling function with detailed logging to track exactly what's happening.

        Args:
            session: The aiohttp ClientSession
            message_id: The ID of the original message with the grid
            task_id: The task ID for this generation
            max_attempts: Maximum number of polling attempts
            delay: Delay between polling attempts in seconds
        """
        app.logger.error(f"[POLL] Starting to poll for upscaled images for task {task_id}")
        app.logger.error(f"[POLL] Original message ID: {message_id}")
        app.logger.error(f"[POLL] Will make {max_attempts} attempts with {delay}s delay")

        # Initialize Redis connection
        try:
            redis = await self.get_async_redis()
            app.logger.error("[POLL] Redis connection initialized")
        except Exception as e:
            app.logger.error(f"[POLL] Error initializing Redis: {e}")
            return False

        # Get metadata about this task
        try:
            metadata_key = f"midjourney_task:{task_id}"
            prompt_bytes = await redis.hget(metadata_key, 'prompt')
            # Fix for the decode error - check type before decoding
            if isinstance(prompt_bytes, bytes):
                prompt = prompt_bytes.decode('utf-8')
            else:
                # It's already a string, use as is
                prompt = prompt_bytes if prompt_bytes else "Unknown prompt"
            app.logger.error(f"[POLL] Task prompt: {prompt[:50]}...")
        except Exception as e:
            app.logger.error(f"[POLL] Error getting prompt from Redis: {e}")
            prompt = "Unknown prompt"

        # Track which variations we've found
        found_variations = set()

        for attempt in range(max_attempts):
            app.logger.error(f"[POLL] Attempt {attempt + 1}/{max_attempts} starting")

            await asyncio.sleep(delay)

            try:
                # Get the latest messages from the channel (increase limit to 50)
                app.logger.error(f"[POLL] Fetching messages from Discord channel {self.channel_id}")
                async with session.get(
                        f"https://discord.com/api/v9/channels/{self.channel_id}/messages?limit=50",
                        timeout=aiohttp.ClientTimeout(total=30)
                ) as msg_response:
                    if msg_response.status != 200:
                        app.logger.error(f"[POLL] Failed to get Discord messages: status={msg_response.status}")
                        continue

                    messages = await msg_response.json()
                    app.logger.error(f"[POLL] Got {len(messages)} messages from Discord")

                    # Debug: Print summary of first few messages
                    for i, message in enumerate(messages[:5]):
                        author = message.get('author', {})
                        content_snippet = message.get('content', '')[:50] + '...' if len(
                            message.get('content', '')) > 50 else message.get('content', '')
                        has_attachments = len(message.get('attachments', [])) > 0
                        app.logger.error(
                            f"[POLL] Message {i}: Author ID: {author.get('id')}, Has attachments: {has_attachments}, Content: {content_snippet}")

                    # Look for upscaled image messages from the Midjourney bot
                    for message in messages:
                        author = message.get('author', {})
                        author_id = author.get('id')
                        content = message.get('content', '')
                        attachments = message.get('attachments', [])

                        # Debug info about this message
                        app.logger.error(
                            f"[POLL] Checking message: Author ID: {author_id}, Midjourney Bot ID: {self.MIDJOURNEY_BOT_ID}")
                        app.logger.error(f"[POLL] Content starts with: {content[:30]}...")
                        app.logger.error(f"[POLL] Has attachments: {len(attachments) > 0}")

                        # Skip if not from Midjourney bot
                        if author_id != self.MIDJOURNEY_BOT_ID:
                            continue

                        # Skip if no attachments
                        if not attachments:
                            app.logger.error(f"[POLL] Skipping message - no attachments")
                            continue

                        # Check if this message contains our prompt and might be an upscale
                        is_upscale = "Upscaled" in content
                        contains_prompt_keywords = False

                        # Check for keywords from the prompt
                        prompt_keywords = [word for word in prompt.split() if len(word) > 5][:3]
                        for keyword in prompt_keywords:
                            if keyword.lower() in content.lower():
                                contains_prompt_keywords = True
                                app.logger.error(f"[POLL] Found prompt keyword: {keyword}")
                                break

                        # Also check if this is a direct response to our original message
                        references_original = False
                        if message.get('referenced_message', {}).get('id') == message_id:
                            references_original = True
                            app.logger.error(f"[POLL] Message references original message")

                        # Check other indicators that this might be an upscale
                        has_upscale_indicator = any(indicator in content for indicator in [
                            "Upscaled by", "Image #", "Upscaled", "Variation",
                            "Upscaling", "Upscale"
                        ])

                        if is_upscale or contains_prompt_keywords or references_original or has_upscale_indicator:
                            app.logger.error(f"[POLL] Potential upscale found!")
                            app.logger.error(f"[POLL] Is upscale: {is_upscale}")
                            app.logger.error(f"[POLL] Contains prompt keywords: {contains_prompt_keywords}")
                            app.logger.error(f"[POLL] References original: {references_original}")
                            app.logger.error(f"[POLL] Has upscale indicator: {has_upscale_indicator}")

                            # Extract variation number from the message content
                            variation_number = None

                            # Try different patterns to find the variation number
                            upscale_patterns = [
                                r'Upscaled by (\d)',
                                r'U(\d)',
                                r'Image #(\d)',
                                r'Variation (\d)'
                            ]

                            for pattern in upscale_patterns:
                                match = re.search(pattern, content)
                                if match:
                                    variation_number = int(match.group(1))
                                    app.logger.error(
                                        f"[POLL] Found variation number {variation_number} using pattern {pattern}")
                                    break

                            # If we can't determine the variation number, generate a unique one
                            if variation_number is None:
                                # Find an unused number
                                for i in range(1, 5):
                                    if i not in found_variations:
                                        variation_number = i
                                        app.logger.error(
                                            f"[POLL] Assigned variation number {variation_number} (not found in content)")
                                        break

                            # Skip if we've already found this variation
                            if variation_number in found_variations:
                                app.logger.error(f"[POLL] Already found variation {variation_number}, skipping")
                                continue

                            # Get the image URL from the first attachment
                            image_url = attachments[0].get('url')
                            app.logger.error(f"[POLL] Image URL: {image_url}")

                            if not image_url:
                                app.logger.error(f"[POLL] No image URL found in attachment, skipping")
                                continue

                            app.logger.error(f"[POLL] Found upscaled image {variation_number} for task {task_id}")

                            # Store this upscaled image in group and individual storage
                            await self.store_upscaled_image(redis, task_id, variation_number, image_url)

                            # Add to set of found variations
                            found_variations.add(variation_number)

                            # If we've found all 4 variations, we're done
                            if len(found_variations) >= 4:
                                app.logger.error(f"[POLL] All upscaled images for task {task_id} found!")
                                return True

                    # Log progress at the end of each attempt
                    app.logger.error(
                        f"[POLL] Found {len(found_variations)}/4 upscaled images after attempt {attempt + 1}")
                    app.logger.error(f"[POLL] Variations found so far: {found_variations}")

                    # If we've found some but not all variations, still update the group status
                    if found_variations and len(found_variations) < 4:
                        app.logger.error(f"[POLL] Updating group with partial results")
                        await self.update_group_status(redis, task_id, "partial")

            except Exception as e:
                app.logger.error(f"[POLL] Error during attempt {attempt + 1}: {str(e)}")
                app.logger.error(f"[POLL] Exception type: {type(e).__name__}")
                import traceback
                app.logger.error(f"[POLL] Traceback: {traceback.format_exc()}")

        # If we get here, we couldn't find all variations within max_attempts
        app.logger.error(
            f"[POLL] Polling complete. Found {len(found_variations)}/4 upscaled images after {max_attempts} attempts")
        if found_variations:
            app.logger.error(f"[POLL] Updating group with partial results before ending")
            await self.update_group_status(redis, task_id, "partial")
        return len(found_variations) > 0

    async def store_upscaled_image(self, redis, task_id, variation_number, image_url):
        """Helper method to store an upscaled image in Redis"""
        try:
            app.logger.error(f"[STORE] Storing upscaled image {variation_number} for task {task_id}")

            # Create image data
            image_data = {
                'url': image_url,
                'variation_number': variation_number,
                'choice': variation_number,  # For frontend compatibility
                'timestamp': datetime.now().isoformat(),
                'key': f"midjourney_image:{task_id}:{variation_number}"
            }

            # Get the group data
            group_key = f"midjourney_group:{task_id}"
            group_data_bytes = await redis.get(group_key)

            if group_data_bytes:
                try:
                    group_info = json.loads(group_data_bytes)
                    app.logger.error(f"[STORE] Got existing group data with {len(group_info.get('images', []))} images")

                    # Check if images array exists
                    if 'images' not in group_info:
                        group_info['images'] = []

                    # Check if this variation already exists
                    existing_index = None
                    for i, img in enumerate(group_info['images']):
                        if img.get('variation_number') == variation_number:
                            existing_index = i
                            break

                    # Add or replace image
                    if existing_index is not None:
                        app.logger.error(f"[STORE] Replacing existing image at index {existing_index}")
                        group_info['images'][existing_index] = image_data
                    else:
                        app.logger.error(f"[STORE] Adding new image to group")
                        group_info['images'].append(image_data)

                    # Update status if all images are received
                    if len(group_info['images']) >= 4:
                        app.logger.error(f"[STORE] All 4 images received, updating status to completed")
                        group_info['status'] = 'completed'
                        metadata_key = f"midjourney_task:{task_id}"
                        await redis.hset(metadata_key, 'status', 'completed'.encode('utf-8'))
                    elif len(group_info['images']) > 0:
                        app.logger.error(
                            f"[STORE] {len(group_info['images'])} images received, updating status to partial")
                        group_info['status'] = 'partial'

                    # Import the TEMP_STORAGE_DURATION
                    from config import TEMP_STORAGE_DURATION

                    # Save group data back to Redis
                    await redis.setex(
                        group_key,
                        TEMP_STORAGE_DURATION,
                        json.dumps(group_info)
                    )

                    # Also store the individual image
                    image_key = f"midjourney_image:{task_id}:{variation_number}"
                    await redis.setex(
                        image_key,
                        TEMP_STORAGE_DURATION,
                        json.dumps(image_data)
                    )

                    app.logger.error(f"[STORE] Successfully stored image {variation_number} for task {task_id}")
                    return True

                except json.JSONDecodeError as e:
                    app.logger.error(f"[STORE] Error parsing group data: {e}")
                    return False
            else:
                app.logger.error(f"[STORE] Group data not found for task {task_id}")
                return False

        except Exception as e:
            app.logger.error(f"[STORE] Error storing upscaled image: {e}")
            return False

    async def update_group_status(self, redis, task_id, status):
        """Helper method to update the status of a group"""
        try:
            app.logger.error(f"[UPDATE] Updating group status to {status} for task {task_id}")

            # Get the group data
            group_key = f"midjourney_group:{task_id}"
            group_data_bytes = await redis.get(group_key)

            if group_data_bytes:
                try:
                    group_info = json.loads(group_data_bytes)

                    # Update status
                    group_info['status'] = status

                    # Import the TEMP_STORAGE_DURATION
                    from config import TEMP_STORAGE_DURATION

                    # Save back to Redis
                    await redis.setex(
                        group_key,
                        int(TEMP_STORAGE_DURATION.total_seconds()),
                        json.dumps(group_info)
                    )

                    # Update task status if needed
                    if status == 'completed':
                        metadata_key = f"midjourney_task:{task_id}"
                        await redis.hset(metadata_key, 'status', 'completed'.encode('utf-8'))

                    app.logger.error(f"[UPDATE] Successfully updated group status to {status}")
                    return True

                except json.JSONDecodeError as e:
                    app.logger.error(f"[UPDATE] Error parsing group data: {e}")
                    return False
            else:
                app.logger.error(f"[UPDATE] Group data not found for task {task_id}")
                return False

        except Exception as e:
            app.logger.error(f"[UPDATE] Error updating group status: {e}")
            return False

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

            # Version redis.asyncio (nouvelle)
            async with redis.pipeline() as pipe:
                pipe.hset(metadata_key, mapping=metadata)  # hmset -> hset avec mapping
                pipe.expire(metadata_key, 3600)
                await pipe.execute()

            # Setup Discord session
            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json"
            }

            timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=30)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                # Get command version
                command_id, command_version = await self.get_application_commands(session)
                app.logger.error(f"Imagine command sent. Awaiting initial response...")
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
                        app.logger.error(f"Got message response: {message is not None}")

                        if message:
                            message_id = message.get('id')
                            components = message.get('components', [])
                            initial_image_url = message.get('attachments', [{}])[0].get('url')
                            app.logger.error(f"Message ID: {message_id}")
                            app.logger.error(f"Components found: {len(components)}")

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
                                        app.logger.error(f"Found upscale button: {button.get('custom_id')}")

                            if upscale_buttons:
                                app.logger.error(f"Found {len(upscale_buttons)} upscale buttons")

                                # Trigger upscales in background
                                upscale_results = await self.handle_upscales(session, message_id, upscale_buttons, task_id)
                                app.logger.error(f"Upscale results: {upscale_results}")

                                # Update Redis with upscale status
                                for result in upscale_results:
                                    if result["success"]:
                                        await redis.hset(metadata_key, f'upscale_status_{result["button"]}',b'triggered')

                                app.logger.error(f"Starting to poll for upscaled images for task {task_id}")
                                polling_result = await self.poll_for_upscaled_images(
                                    session=session,
                                    message_id=message_id,
                                    task_id=task_id,
                                    max_attempts=30,  # Increase attempts for better chances
                                    delay=10  # 10 seconds between attempts
                                )
                                app.logger.error(f"Polling completed with result: {polling_result}")

                                # We run this in a background task so the current request can return immediately
                                app.logger.error(
                                    f"Starting background task to poll for upscaled images for task {task_id}")
                                asyncio.create_task(
                                    self.poll_for_upscaled_images(
                                        session=session,
                                        message_id=message_id,
                                        task_id=task_id,
                                        max_attempts=30,
                                        delay=10
                                    )
                                )
                            return {
                                "success": True,
                                "status": "processing",
                                "task_id": task_id
                            }

                        else:
                            await redis.hset(metadata_key, 'status', b'error')
                            await redis.hset(metadata_key, 'error', b'No response from Midjourney')
                            return {
                                "success": False,
                                "error": "No response from Midjourney"
                            }
                    else:
                        await redis.hset(metadata_key, 'status', b'error')
                        await redis.hset(metadata_key, 'error', b'Failed to send command to Discord')
                        return {
                            "success": False,
                            "error": "Failed to send command to Discord"
                        }

        except Exception as e:
            # Clean up Redis data in case of error
            app.logger.error(f"Error in generate: {str(e)}")
            if metadata_key:
                try:
                    redis = await self.get_async_redis()
                    await redis.hset(metadata_key, 'status', b'error')
                    await redis.hset(metadata_key, 'error', str(e).encode('utf-8'))
                except:
                    pass

                return {"success": False, "error": str(e)}

        finally:
            # Close Redis connection if it exists
            if self.async_redis is not None:
                await self.async_redis.close()
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
            # Create a task with timeout to prevent hanging
            coro = self.generate_image(model_name, prompt, additional_params)
            task = asyncio.ensure_future(coro, loop=loop)
            return loop.run_until_complete(asyncio.wait_for(task, timeout=180))
        finally:
            # Cancel any pending tasks before closing
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            # Run cancellation of tasks
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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

