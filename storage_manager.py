# storage_manager.py
import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from flask import current_app as app
from io import BytesIO

from gallery_manager import logger

logger = logging.getLogger("storage_manager")

# Mode test et valeurs par défaut
TEST_MODE = os.getenv("STORAGE_TEST_MODE", "false").lower() == "true"
TEST_TTL_MINUTES = int(os.getenv("STORAGE_TEST_TTL_MINUTES", "10"))

class FileStorage:
    """Gestionnaire de stockage sur système de fichiers"""

    def __init__(self, base_path):
        """Initialiser le stockage avec un chemin de base"""
        self.base_path = Path(base_path)
        self.ensure_directories()

    def ensure_directories(self):
        """S'assurer que les répertoires nécessaires existent"""
        for content_type in ['images', 'audio', 'video', 'metadata']:
            path = self.base_path / content_type
            path.mkdir(parents=True, exist_ok=True)

    def generate_path(self, key, content_type):
        """Générer un chemin pour stocker un fichier basé sur sa clé"""
        # Utiliser le hash de la clé pour créer un système de sous-répertoires
        # Cela empêche d'avoir trop de fichiers dans un seul répertoire
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        subdir = key_hash[:2] + '/' + key_hash[2:4]

        # Créer le sous-répertoire s'il n'existe pas
        path = self.base_path / content_type / subdir
        path.mkdir(parents=True, exist_ok=True)

        return path / key_hash

    def store_file(self, key, data, content_type):
        """Stocker des données binaires dans un fichier"""
        file_path = self.generate_path(key, content_type)

        with open(file_path, 'wb') as f:
            f.write(data)

        return str(file_path)

    def store_metadata(self, key, metadata):
        """Stocker les métadonnées en JSON"""
        metadata_path = self.generate_path(key, 'metadata')

        # Ajouter un timestamp pour le moment où le fichier a été déplacé sur disque
        metadata['disk_storage_date'] = datetime.now().isoformat()

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return str(metadata_path)

    def retrieve_file(self, key, content_type):
        """Lire des données binaires depuis un fichier"""
        file_path = self.generate_path(key, content_type)

        if not file_path.exists():
            return None

        with open(file_path, 'rb') as f:
            return f.read()

    def retrieve_metadata(self, key):
        """Lire les métadonnées depuis un fichier JSON"""
        metadata_path = self.generate_path(key, 'metadata')

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def delete_file(self, key, content_type):
        """Supprimer un fichier"""
        file_path = self.generate_path(key, content_type)

        if file_path.exists():
            os.remove(file_path)
            return True
        return False

    def delete_metadata(self, key):
        """Supprimer un fichier de métadonnées"""
        metadata_path = self.generate_path(key, 'metadata')

        if metadata_path.exists():
            os.remove(metadata_path)
            return True
        return False


class StorageManager:
    """Gestionnaire de stockage unifié (Redis + système de fichiers)"""

    def __init__(self, redis_client, file_storage, temp_duration=None):
        """Initialiser le gestionnaire avec Redis et FileStorage"""
        self.redis = redis_client
        self.file_storage = file_storage

        # Détermination automatique du TTL en fonction du mode (test ou prod)
        if temp_duration is None:
            if TEST_MODE:
                self.temp_duration = timedelta(minutes=TEST_TTL_MINUTES)
                print(f"Mode TEST: TTL auto-configuré à {TEST_TTL_MINUTES} minutes")
            else:
                self.temp_duration = timedelta(hours=24)
                print("Mode PROD: TTL standard de 24 heures")
        else:
            self.temp_duration = temp_duration

        # Log du TTL configuré
        print(f"StorageManager initialisé avec TTL: {self.temp_duration}")
        logger.info(f"StorageManager initialisé avec TTL: {self.temp_duration}")

    def store(self, key, data, metadata=None, content_type='images'):
        """Stocker le contenu dans Redis avec métadonnées optionnelles"""

        # Force du TTL en mode TEST
        ttl_seconds = int(self.temp_duration.total_seconds())


        logger.error(f"Stockage de {key} avec TTL: {ttl_seconds}")
        logger.info(f"Mode TEST activé: {TEST_MODE}")

        # Stocker les données dans Redis
        self.redis.setex(
            key,
            ttl_seconds,
            data
        )

        #Preparation metadata
        if metadata is None:
            metadata = {}

        # NOUVEAU: Ajouter des informations sur le stockage dans les métadonnées
        metadata['storage_timestamp'] = datetime.now().isoformat()
        metadata['storage_ttl'] = ttl_seconds
        metadata['storage_test_mode'] = TEST_MODE

        # Stocker les métadonnées si fournies

        metadata_key = f"{key}:meta"
        if isinstance(metadata, dict):
            # Convertir dict en format Redis
            formatted_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, str):
                    formatted_metadata[k] = v.encode('utf-8')
                else:
                    formatted_metadata[k] = v
            self.redis.hmset(metadata_key, formatted_metadata)
        else:
            # Supposer que les métadonnées sont déjà au format Redis
            self.redis.hmset(metadata_key, metadata)

        self.redis.expire(metadata_key, int(self.temp_duration.total_seconds()))
        return key

    def get(self, key, content_type='images'):
        """Récupérer le contenu depuis Redis ou le système de fichiers"""
        # Essayer d'abord Redis
        data = self.redis.get(key)

        # Si pas dans Redis, essayer le stockage fichier
        if data is None:
            logger.info(f"Contenu {key} non trouvé dans Redis, vérification du stockage fichier")
            data = self.file_storage.retrieve_file(key, content_type)

        return data

    def get_metadata(self, key):
        """Récupérer les métadonnées depuis Redis ou le système de fichiers"""
        # Essayer d'abord Redis
        metadata_key = f"{key}:meta"
        redis_metadata = self.redis.hgetall(metadata_key)

        if redis_metadata:
            # Convertir les bytes Redis en dict
            return {k.decode('utf-8') if isinstance(k, bytes) else k:
                        v.decode('utf-8') if isinstance(v, bytes) else v
                    for k, v in redis_metadata.items()}

        # Si pas dans Redis, essayer le stockage fichier
        logger.info(f"Métadonnées pour {key} non trouvées dans Redis, vérification du stockage fichier")
        return self.file_storage.retrieve_metadata(key)

    def migrate_to_disk(self, key, content_type='images'):
        """Migrer le contenu de Redis vers le stockage fichier"""
        # Récupérer les données et métadonnées depuis Redis
        data = self.redis.get(key)
        metadata_key = f"{key}:meta"
        redis_metadata = self.redis.hgetall(metadata_key)

        if data is None:
            logger.warning(f"Impossible de migrer {key}: non trouvé dans Redis")
            return False

        # Convertir le format de métadonnées Redis en dict
        metadata = {
            k.decode('utf-8') if isinstance(k, bytes) else k:
                v.decode('utf-8') if isinstance(v, bytes) else v
            for k, v in redis_metadata.items()
        } if redis_metadata else {}

        # Ajouter des infos de stockage aux métadonnées
        metadata['storage_migration_date'] = datetime.now().isoformat()
        metadata['original_redis_key'] = key

        # Stocker dans le système de fichiers
        file_path = self.file_storage.store_file(key, data, content_type)
        metadata_path = self.file_storage.store_metadata(key, metadata)

        logger.info(f"Migré {key} vers le disque: {file_path}")

        # Supprimer de Redis après migration réussie
        self.redis.delete(key)
        self.redis.delete(metadata_key)

        return True

    def delete(self, key, content_type='images'):
        """Supprimer le contenu de Redis et du stockage fichier"""
        # Supprimer de Redis
        redis_deleted = self.redis.delete(key) > 0
        metadata_deleted = self.redis.delete(f"{key}:meta") > 0

        # Supprimer du stockage fichier
        file_deleted = self.file_storage.delete_file(key, content_type)
        metadata_file_deleted = self.file_storage.delete_metadata(key)

        return redis_deleted or file_deleted