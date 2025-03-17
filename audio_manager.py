from datetime import datetime
from flask import current_app as app
import json
import redis
import os
from io import BytesIO

from config import TEMP_STORAGE_DURATION

class AudioManager:
    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.prefix = "audio"

    def _generate_key(self, user_id, type="temp"):
        """Génère une clé unique pour un audio"""
        timestamp = datetime.now().timestamp()
        return f"{self.prefix}:{user_id}:{timestamp}"

    def store_temp_audio(self, user_id, audio_data, metadata=None):
        """Stocke un fichier audio temporaire avec ses métadonnées"""
        try:
            # Génération d'une clé unique
            key = self._generate_key(user_id)
            app.logger.error(f"Generated key for audio: {key}")

            # Convertir les métadonnées dict en format Redis si nécessaire
            if metadata and isinstance(metadata, dict):
                redis_metadata = {
                    k: v.encode('utf-8') if isinstance(v, str) else v
                    for k, v in metadata.items()
                }
            else:
                redis_metadata = metadata

            # Stocker via le gestionnaire de stockage
            self.storage.store(key, audio_data, redis_metadata, 'audio')

            # Ajouter à l'index utilisateur
            user_index_key = f"user:{user_id}:audio"
            self.storage.redis.lpush(user_index_key, key)

            return key

        except Exception as e:
            app.logger.error(f"Storage error: {str(e)}")
            app.logger.error(f"Failed to store audio with metadata: {metadata}")
            raise


    def get_user_history(self, user_id, limit=100):
        """Récupère l'historique des audios d'un utilisateur"""
        app.logger.error(f"Fetching audio history for user {user_id}")

        pattern = f"{self.prefix}:{user_id}:*"

        # Récupérer toutes les clés qui correspondent à ce pattern
        all_keys = self.storage.redis.keys(pattern)
        app.logger.error(f"Found {len(all_keys)} audio keys")

        # Filtrer pour ne garder que les clés d'audio (pas les métadonnées)
        audio_keys = [k for k in all_keys if not k.endswith(b':meta')]

        # Préparer la liste des entrées d'historique
        history = []

        for key in audio_keys:
            try:
                # Convertir la clé en string si nécessaire
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key

                # Récupérer les métadonnées via le StorageManager
                metadata = self.storage.get_metadata(key_str)

                # Vérifier si l'audio existe toujours
                if not self.storage.get(key_str, 'audio'):
                    app.logger.error(f"Audio {key_str} no longer exists, skipping")
                    continue

                # Préparer l'entrée d'historique
                entry = {
                    'key': key_str,
                    'url': f"/audio/{key_str}",
                    'timestamp': metadata.get('timestamp', '') if metadata else '',
                    'voice': metadata.get('voice', 'default') if metadata else 'default',
                    'text': metadata.get('text', '') if metadata else '',
                    'type': 'audio'
                }

                # Ajouter à l'historique
                history.append(entry)

            except Exception as e:
                app.logger.error(f"Error processing audio key {key}: {str(e)}")
                continue

        # Trier par date (plus récent en premier) et limiter le nombre d'entrées
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        return sorted_history

    def get_audio(self, key):
        """Récupère un fichier audio depuis n'importe quel stockage"""
        return self.storage.get(key, 'audio')

    def delete_audio(self, key):
        """Suppression propre avec ses métadonnées"""
        return self.storage.delete(key, 'audio')

    def get_audio_metadata(self, key):
        """Récupère les métadonnées d'un audio"""
        return self.storage.get_metadata(key)