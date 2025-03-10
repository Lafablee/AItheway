from datetime import datetime
from flask import current_app as app
import json
import redis
import os
from io import BytesIO

from config import TEMP_STORAGE_DURATION

class AudioManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "audio"

    def _generate_key(self, user_id, type="temp"):
        """Génère une clé unique avec préfixe"""
        timestamp = datetime.now().timestamp()
        return f"audio:temp:file:{user_id}:{timestamp}"

    def store_temp_audio(self, user_id, audio_data, metadata=None):
        """Stocke un fichier audio temporaire avec ses métadonnées"""
        try:
            # Génération d'une clé unique
            key = self._generate_key(user_id)
            app.logger.error(f"Generated key for audio: {key}")

            # Pipeline pour les opérations atomiques
            pipe = self.redis.pipeline()

            # Stockage de l'audio
            pipe.setex(
                key,
                TEMP_STORAGE_DURATION,
                audio_data
            )

            # Stockage des métadonnées si présentes
            if metadata:
                metadata_key = f"{key}:meta"
                app.logger.error(f"Original metadata: {metadata}")

                pipe.hmset(metadata_key, metadata)
                pipe.expire(metadata_key, TEMP_STORAGE_DURATION)

                # Ajouter à l'index utilisateur
                user_index_key = f"user:{user_id}:audios"
                pipe.lpush(user_index_key, key)

            results = pipe.execute()
            app.logger.error(f"Pipeline execution results: {results}")

            return key

        except Exception as e:
            app.logger.error(f"Redis storage error: {str(e)}")
            app.logger.error(f"Failed to store audio with metadata: {metadata}")
            raise

    def get_user_history(self, user_id):
        """Récupère l'historique des audios d'un utilisateur"""
        app.logger.error(f"Fetching audio history for user {user_id}")

        patterns = [f"{self.prefix}:temp:file:{user_id}:*"]
        audios = []

        all_keys = []
        for pattern in patterns:
            keys = self.redis.keys(pattern)
            all_keys.extend(keys)

        app.logger.error(f"Found all keys: {all_keys}")

        # Filtrer pour ne garder que les clés d'audio (pas les métadonnées)
        audio_keys = [k for k in all_keys if not k.endswith(b':meta')]
        app.logger.error(f"Filtered audio keys: {audio_keys}")

        for key in audio_keys:
            try:
                # Construire la clé des métadonnées
                metadata_key = f"{key.decode('utf-8')}:meta"
                metadata = self.redis.hgetall(metadata_key)
                app.logger.error(f"Checking metadata for key {metadata_key}: {metadata}")

                # Vérifier si l'audio existe toujours
                if not self.redis.exists(key):
                    app.logger.error(f"Audio {key} no longer exists, skipping")
                    continue

                try:
                    # Décoder les métadonnées essentielles
                    decoded_text = metadata.get(b'text', b'').decode('utf-8')
                    decoded_timestamp = metadata.get(b'timestamp', b'').decode('utf-8')
                    voice = metadata.get(b'voice', b'alloy').decode('utf-8')

                    # Construire l'objet audio
                    audio_data = {
                        'key': key.decode('utf-8'),
                        'url': f"/audio/{key.decode('utf-8')}",
                        'text': decoded_text,
                        'timestamp': decoded_timestamp,
                        'voice': voice
                    }

                    # Ajouter la vitesse si présente
                    if b'speed' in metadata:
                        audio_data['speed'] = float(metadata.get(b'speed').decode('utf-8'))

                    audios.append(audio_data)
                    app.logger.error(f"Added audio to history: {audio_data}")

                except Exception as e:
                    app.logger.error(f"Error processing audio metadata: {e}")
                    continue

            except Exception as e:
                app.logger.error(f"Error processing key {key}: {e}")
                continue

        # Trier les audios par timestamp (plus récent en premier)
        sorted_audios = sorted(audios, key=lambda x: x['timestamp'], reverse=True)
        app.logger.error(f"Returning {len(sorted_audios)} sorted audios")

        return sorted_audios

    def get_audio(self, key):
        """Récupère un fichier audio"""
        if not self.redis.exists(key):
            return None
        return self.redis.get(key)

    def delete_audio(self, key):
        """Suppression propre avec métadonnées"""
        pipe = self.redis.pipeline()
        pipe.delete(key)
        pipe.delete(f"{key}:meta")
        pipe.execute()

    def get_audio_metadata(self, key):
        """Récupère les métadonnées d'un audio"""
        metadata_key = f"{key}:meta"
        return self.redis.hgetall(metadata_key)