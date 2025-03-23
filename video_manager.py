import os
import json
import traceback

import requests
import time
from datetime import datetime, timedelta
from io import BytesIO
from flask import current_app as app


class VideoManager:
    """
    Gestionnaire de vidéos généré par MiniMax.
    """

    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.prefix = "video"

    def _generate_key(self, user_id, type="temp"):
        """Génère une clé unique avec préfixe"""
        timestamp = datetime.now().timestamp()
        return f"{self.prefix}:{type}:{user_id}:{timestamp}"

    def store_video_metadata(self, user_id, task_id, prompt, model, additional_params=None):
        """
        Stocke les métadonnées d'une vidéo en cours de génération.
        Retourne la clé de la vidéo.
        """
        try:
            # Génère une clé unique pour cette vidéo
            video_key = self._generate_key(user_id)
            app.logger.error(f"Generated video_key: {video_key} for task_id: {task_id}")

            # Prépare les métadonnées
            metadata = {
                'type': 'video',
                'task_id': task_id,
                'prompt': prompt,
                'model': model,
                'status': 'processing',
                'timestamp': datetime.now().isoformat(),
                'user_id': str(user_id)
            }

            # Ajoute les paramètres additionnels aux métadonnées
            if additional_params:
                if isinstance(additional_params, str):
                    try:
                        additional_params = json.loads(additional_params)
                    except json.JSONDecodeError:
                        additional_params = {}
                metadata['parameters'] = json.dumps(additional_params)

            # Convertir les métadonnées en format Redis
            redis_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, str):
                    redis_metadata[k] = v.encode("utf-8")
                elif isinstance(v, bool):
                    #Convertir les booléens en chaînes en "true" ou "false"
                    redis_metadata[k] = str(v).lower().encode("utf-8")
                elif v is None:
                    # Gérer els valerus None
                    redis_metadata[k] = b''
                else:
                    #pour les autres types (int, float, bytes)
                    redis_metadata[k] = v

            # Stocker via le gestionnaire de stockage
            # Comme nous n'avons pas encore de données vidéo, on passe None
            self.storage.store(video_key, None, redis_metadata, 'videos')

            # Ajouter le mapping task_id -> video_key
            task_mapping_key = f"video:task:{task_id}"
            app.logger.error(f"Setting mapping: {task_mapping_key} -> {video_key}")
            self.storage.redis.set(task_mapping_key, video_key)

            # Ajouter à l'index utilisateur
            user_index_key = f"user:{user_id}:videos"
            self.storage.redis.lpush(user_index_key, video_key)
            self.storage.redis.expire(user_index_key, 30 * 24 * 60 * 60)  # 30 jours

            # Expirer le mapping après 7 jours
            self.storage.redis.expire(task_mapping_key, 7 * 24 * 60 * 60)

            return video_key

        except Exception as e:
            app.logger.error(f"Error storing video metadata: {str(e)}")
            app.logger.error(traceback.format_exc())
            return None

    def update_video_status(self, task_id, status, file_id=None, download_url=None):
        """
        Met à jour le statut d'une vidéo à partir de son task_id.
        """
        try:
            # Récupère la clé vidéo à partir du task_id
            task_mapping_key = f"video:task:{task_id}"
            video_key = self.storage.redis.get(task_mapping_key)

            if not video_key:
                app.logger.error(f"No video key found for task_id {task_id}")
                return False

            # Convertit en chaîne si c'est un bytes
            if isinstance(video_key, bytes):
                video_key = video_key.decode('utf-8')

            # Récupère les métadonnées existantes
            metadata = self.storage.get_metadata(video_key)
            if not metadata:
                app.logger.error(f"No metadata found for video key {video_key}")
                return False

            # Met à jour les métadonnées
            metadata['status'] = status
            if file_id:
                metadata['file_id'] = file_id
            if download_url:
                metadata['download_url'] = download_url

            # Stocke les métadonnées mises à jour
            self.storage.store_metadata(video_key, metadata)

            app.logger.info(f"Updated video status for task {task_id} to {status}")
            return True

        except Exception as e:
            app.logger.error(f"Error updating video status: {str(e)}")
            return False

    def store_video_file(self, video_key, video_data):
        """
        Stocke le fichier vidéo lui-même.
        """
        try:
            # Stocke le fichier vidéo
            self.storage.store(video_key, video_data, content_type='videos')

            # Met à jour les métadonnées pour indiquer que le fichier est stocké
            metadata = self.storage.get_metadata(video_key)
            if metadata:
                metadata['file_stored'] = True
                self.storage.store_metadata(video_key, metadata)

            app.logger.info(f"Stored video file for key {video_key}")
            return True

        except Exception as e:
            app.logger.error(f"Error storing video file: {str(e)}")
            return False

    def get_video_by_task_id(self, task_id):
        """
        Récupère les informations d'une vidéo à partir de son task_id.
        """
        try:
            app.logger.error(f"Atempting to get video by task_id {task_id}")
            # Récupère la clé vidéo à partir du task_id
            task_mapping_key = f"video:task:{task_id}"


            if not self.storage.redis.exists(task_mapping_key):
                app.logger.error(f"No video key found for task_id {task_id}")
                return None

            video_key = self.storage.redis.get(task_mapping_key)

            # Convertit en chaîne si c'est un bytes
            if isinstance(video_key, bytes):
                video_key = video_key.decode('utf-8')

            app.logger.info(f"Found video key: {video_key} for task_id:  {task_id}")

            # Récupère les métadonnées
            metadata = self.storage.get_metadata(video_key)
            if not metadata:
                app.logger.error(f"No metadata found for video key {video_key}")
                return None

            # Ajoute la clé vidéo aux métadonnées pour faciliter l'utilisation
            metadata['video_key'] = video_key

            return metadata

        except Exception as e:
            app.logger.error(f"Error getting video by task_id: {str(e)}")
            return None

    def get_video(self, video_key):
        """
        Récupère le fichier vidéo.
        """
        try:
            # Récupère le fichier vidéo
            video_data = self.storage.get(video_key, content_type='videos')

            if not video_data:
                app.logger.error(f"No video data found for key {video_key}")
                return None

            return video_data

        except Exception as e:
            app.logger.error(f"Error getting video: {str(e)}")
            return None

    def download_from_url(self, url):
        """
        Télécharge une vidéo à partir d'une URL externe, avec gestion des erreurs 403.
        """
        try:
            response = requests.get(url, stream=True, timeout=60)

            # Si l'URL est expirée (403 Forbidden), tenter de rafraîchir l'URL
            if response.status_code == 403:
                app.logger.warning(f"URL expirée (403): {url}")

                # Extraire le task_id depuis l'URL ou les métadonnées
                task_id = None

                # Chercher les métadonnées qui correspondent à cette URL
                for key in self.storage.redis.keys("video:temp:*:meta"):
                    meta_key = key.decode('utf-8') if isinstance(key, bytes) else key
                    video_key = meta_key.replace(':meta', '')
                    metadata = self.storage.get_metadata(video_key)

                    if metadata and metadata.get('download_url') == url and metadata.get('task_id'):
                        task_id = metadata.get('task_id')
                        app.logger.info(f"Trouvé task_id {task_id} pour l'URL expirée")
                        break

                # Si on a trouvé un task_id, tenter de rafraîchir l'URL via l'API
                if task_id:
                    app.logger.info(f"Tentative de rafraîchissement de l'URL pour task_id {task_id}")

                    # Récupérer le générateur MiniMax depuis ai_manager
                    from ai_models import create_ai_manager
                    ai_manager = create_ai_manager()
                    minimax_video_generator = ai_manager.generators.get("minimax-video")

                    if minimax_video_generator:
                        # Vérifier le statut pour obtenir le file_id
                        status_response = minimax_video_generator.generator.check_generation_status(task_id)

                        if status_response.get('success') and status_response.get('file_id'):
                            file_id = status_response.get('file_id')

                            # Obtenir une nouvelle URL de téléchargement
                            new_url = minimax_video_generator.generator.get_download_url(file_id)

                            if new_url:
                                app.logger.info(f"URL rafraîchie avec succès: {new_url}")

                                # Mettre à jour les métadonnées avec la nouvelle URL
                                if video_key:
                                    metadata['download_url'] = new_url
                                    self.storage.store_metadata(video_key, metadata)

                                # Tenter le téléchargement avec la nouvelle URL
                                return self.download_from_url(new_url)

                app.logger.error(f"Impossible de rafraîchir l'URL expirée: {url}")
                return None

            response.raise_for_status()

            app.logger.info(f"Downloaded video from {url}, size: {len(response.content)} bytes")
            return response.content

        except Exception as e:
            app.logger.error(f"Error downloading video from URL {url}: {str(e)}")
            return None

    def get_user_video_history(self, user_id, page=1, per_page=10):
        """
        Récupère l'historique des vidéos d'un utilisateur.
        """
        try:
            # Récupère les clés des vidéos de l'utilisateur
            user_index_key = f"user:{user_id}:videos"
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page - 1

            # Récupère les clés avec pagination
            video_keys = self.storage.redis.lrange(user_index_key, start_idx, end_idx)

            if not video_keys:
                return []

            # Pour chaque clé, récupère les métadonnées
            videos = []
            for key_bytes in video_keys:
                key = key_bytes.decode('utf-8') if isinstance(key_bytes, bytes) else key_bytes
                metadata = self.storage.get_metadata(key)

                if metadata:
                    # Ajoute la clé vidéo aux métadonnées
                    metadata['video_key'] = key

                    # Ajoute l'URL de la vidéo
                    metadata['video_url'] = f"/video/{key}"

                    videos.append(metadata)

            # Trie par date (les plus récentes d'abord)
            videos.sort(key=lambda v: v.get('timestamp', ''), reverse=True)

            return videos

        except Exception as e:
            app.logger.error(f"Error getting user video history: {str(e)}")
            return []

    def retry_download(self, task_id):
        """
        Retente le téléchargement d'une vidéo à partir de son URL distante.
        """
        try:
            # Récupère les informations de la vidéo
            video_info = self.get_video_by_task_id(task_id)

            if not video_info:
                return False

            # Vérifie si l'URL de téléchargement est disponible
            if not video_info.get('download_url'):
                app.logger.error(f"No download URL available for task_id {task_id}")
                return False

            # Télécharge la vidéo
            video_data = self.download_from_url(video_info['download_url'])

            if not video_data:
                return False

            # Stocke la vidéo
            return self.store_video_file(video_info['video_key'], video_data)

        except Exception as e:
            app.logger.error(f"Error retrying download: {str(e)}")
            return False