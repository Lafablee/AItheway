# storage_manager.py
import os
import json
import hashlib
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import logging
from io import BytesIO
from sys import exception

logger = logging.getLogger("storage_manager")

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
        """Lire des données binaires depuis un fichier avec gestion des erreurs améliorée"""
        try:
            file_path = self.generate_path(key, content_type)

            if not file_path.exists():
                # Essayer avec différentes variations de la clé si nécessaire
                # Par exemple, essayer avec/sans certains préfixes courants
                alternative_keys = [key]

                # Ajout de variantes de clés potentielles pour les cas particuliers
                if ':' in key:
                    # Essayer avec seulement la partie après le dernier ":"
                    parts = key.split(':')
                    if len(parts) > 2:
                        alternative_keys.append(parts[-1])  # Juste l'identifiant
                        alternative_keys.append(f"{parts[0]}:{parts[-1]}")  # Type + ID

                # Essayer chaque variante
                for alt_key in alternative_keys:
                    if alt_key == key:
                        continue  # Déjà essayé

                    alt_path = self.generate_path(alt_key, content_type)
                    if alt_path.exists():
                        file_path = alt_path
                        logger.info(f"Fichier trouvé avec clé alternative: {alt_key}")
                        break

                # Si toujours pas trouvé, retourner None
                if not file_path.exists():
                    return None

            with open(file_path, 'rb') as f:
                return f.read()

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du fichier {key}: {str(e)}")
            return None

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

    def __init__(self, redis_client, file_storage, temp_duration=timedelta(days=1)):
        """Initialiser le gestionnaire avec Redis et FileStorage"""
        self.redis = redis_client
        self.file_storage = file_storage
        self.temp_duration = temp_duration

    def store(self, key, data, metadata=None, content_type='images'):
        """Stocker le contenu dans Redis avec métadonnées optionnelles"""
        # Stocker les données dans Redis
        try:
            if data is not None:
                self.redis.setex(
                    # si on a des donnés on les stock normalement
                    key,
                    int(self.temp_duration.total_seconds()),
                    data
                )
            else:
                # Si data est None, stocker une chaîne vide comme placeholder
                # Ceci est crucial car Redis ne peut pas stocker None directement
                self.redis.setex(
                    key,
                    int(self.temp_duration.total_seconds()),
                    b''  # Bytes vides comme placeholder
                )

            # Stockage des métadonnées - CORRIGE INDENTATION (n'est plus sous le bloc else)
            if metadata:
                # Conversion des métadonnées pour compatibilité Redis
                redis_safe_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, str):
                        redis_safe_metadata[k] = v.encode('utf-8')
                    elif isinstance(v, bool):
                        # Conversion des booléens en chaînes
                        redis_safe_metadata[k] = str(v).lower().encode('utf-8')
                    elif v is None:
                        # Conversion de None en chaîne vide
                        redis_safe_metadata[k] = b''
                    else:
                        # Pour les autres types (int, float, bytes)
                        redis_safe_metadata[k] = v

                # Stocker les métadonnées avec hmset
                metadata_key = f"{key}:meta"
                self.redis.hmset(metadata_key, redis_safe_metadata)
                self.redis.expire(metadata_key, int(self.temp_duration.total_seconds()))

                # Pour faciliter le suivi du type de contenu
                content_type_key = f"content:{content_type}:{key}"
                self.redis.setex(
                    content_type_key,
                    int(self.temp_duration.total_seconds()),
                    key.encode('utf-8')
                )

            return key

        except Exception as e:
            logger.error(f"Error in storage.store: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def store_metadata(self, key, metadata):
        """
        Stocke uniquement les métadonnées pour une clé existante.

        Args:
            key (str): Clé identifiant les données
            metadata (dict): Métadonnées à stocker

        Returns:
            bool: True si réussi, False sinon
        """
        try:
            if not metadata:
                return True  # Rien à stocker

            # Création de la clé de métadonnées
            metadata_key = f"{key}:meta"

            # Conversion des métadonnées pour compatibilité Redis
            redis_safe_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, str):
                    redis_safe_metadata[k] = v.encode('utf-8')
                elif isinstance(v, bool):
                    # Conversion des booléens en chaînes
                    redis_safe_metadata[k] = str(v).lower().encode('utf-8')
                elif v is None:
                    # Conversion de None en chaîne vide
                    redis_safe_metadata[k] = b''
                else:
                    # Pour les autres types (int, float, bytes)
                    redis_safe_metadata[k] = v

            # Stocker les métadonnées
            self.redis.hmset(metadata_key, redis_safe_metadata)
            self.redis.expire(metadata_key, int(self.temp_duration.total_seconds()))

            # Enregistrer aussi sur disque si souhaité
            if hasattr(self, 'file_storage') and self.file_storage:
                try:
                    # Convertir en format standard pour le stockage fichier
                    disk_metadata = {
                        k.decode('utf-8') if isinstance(k, bytes) else k:
                            v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in redis_safe_metadata.items()
                    }
                    self.file_storage.store_metadata(key, disk_metadata)
                except Exception as e:
                    logger.warning(f"Échec de sauvegarde des métadonnées sur disque: {str(e)}")

            return True

        except Exception as e:
            logger.error(f"Error in storage.store_metadata: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def exists(self, key, content_type='images'):
        """Vérifie si une clé existe dans Redis ou dans le stockage fichier"""
        # Vérifier dans Redis
        if self.redis.exists(key):
            return True

        # Vérifier dans le stockage fichier
        file_path = self.file_storage.generate_path(key, content_type)
        return file_path.exists()

    def get(self, key, content_type='images'):
        """Récupérer le contenu depuis Redis ou le système de fichiers de manière plus robuste"""
        # Essayer d'abord Redis
        data = self.redis.get(key)

        # Si le contenu n'est pas dans Redis, essayer le stockage fichier
        if data is None:
            logger.info(f"Contenu {key} non trouvé dans Redis, vérification du stockage fichier")

            # Essayer de récupérer directement
            data = self.file_storage.retrieve_file(key, content_type)

            # Si toujours pas trouvé, essayer une recherche par métadonnées
            if data is None:
                # Récupérer les métadonnées pour ce type de contenu
                metadata = self.get_metadata(key)

                if metadata:
                    logger.info(f"Métadonnées trouvées pour {key}, tentative de récupération via chemin alternatif")

                    # Si c'est une vidéo avec une URL de téléchargement, essayer de la retélécharger
                    if content_type == 'videos' and 'download_url' in metadata:
                        try:
                            from video_manager import VideoManager
                            video_manager = VideoManager(self)
                            data = video_manager.download_from_url(metadata['download_url'])

                            if data:
                                # Stocker pour usage futur
                                self.file_storage.store_file(key, data, content_type)
                                logger.info(f"Retéléchargement et stockage réussis de la vidéo {key}")
                        except Exception as e:
                            logger.error(f"Erreur lors du retéléchargement de la vidéo {key}: {str(e)}")

                # Si toujours pas trouvé et que c'est une image ou un audio, essayer des variantes de clés
                if data is None and content_type in ['images', 'audio']:
                    # Essayer de trouver par le hash MD5 de la clé originale
                    try:
                        import hashlib
                        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()

                        # Chercher dans le répertoire approprié
                        base_path = self.file_storage.base_path
                        type_path = os.path.join(base_path, content_type)

                        # Parcourir les sous-répertoires en fonction du hash
                        for root, dirs, files in os.walk(type_path):
                            if key_hash in files:
                                file_path = os.path.join(root, key_hash)
                                with open(file_path, 'rb') as f:
                                    data = f.read()
                                    logger.info(f"Fichier trouvé via hash MD5: {file_path}")
                                    break
                    except Exception as e:
                        logger.error(f"Erreur lors de la recherche par hash: {str(e)}")

        elif data == b'':
            # Si le contenu est une chaîne vide, considérer comme None
            data = None

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
        """Migrer le contenu de Redis vers le stockage fichier avec des vérifications améliorées"""
        # Récupérer les données et métadonnées depuis Redis
        data = self.redis.get(key)
        metadata_key = f"{key}:meta"
        redis_metadata = self.redis.hgetall(metadata_key)

        if data is None and content_type == 'videos':
            # Cas spécial pour les vidéos: elles peuvent avoir des métadonnées mais aucun contenu dans Redis
            # car elles sont téléchargées à la demande depuis l'URL distante
            if redis_metadata:
                # Convertir le format de métadonnées Redis en dict
                metadata = {
                    k.decode('utf-8') if isinstance(k, bytes) else k:
                        v.decode('utf-8') if isinstance(v, bytes) else v
                    for k, v in redis_metadata.items()
                } if redis_metadata else {}

                # Ne migrer que les métadonnées
                metadata['storage_migration_date'] = datetime.now().isoformat()
                metadata['original_redis_key'] = key

                # Extraire l'ID utilisateur si possible
                try:
                    parts = key.split(':')
                    if len(parts) >= 3:
                        metadata['user_id'] = parts[2]  # video:temp:user_id:timestamp
                except Exception as e:
                    logger.warning(f"Impossible d'extraire l'ID utilisateur de la clé {key}: {str(e)}")

                # Stocker uniquement les métadonnées
                metadata_path = self.file_storage.store_metadata(key, metadata)
                logger.info(f"Migré métadonnées de {key} vers le disque: {metadata_path}")

                # Supprimer métadonnées de Redis après migration réussie
                self.redis.delete(metadata_key)

                return True

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

        # Extraire l'ID utilisateur de la clé (pour une récupération plus facile)
        # Format attendu: "audio:user_id:timestamp" ou "img:temp:image:user_id:timestamp"
        try:
            parts = key.split(':')
            if content_type == 'audio' and len(parts) >= 2:
                metadata['user_id'] = parts[1]  # audio:user_id:timestamp
            elif content_type == 'images' and len(parts) >= 4:
                metadata['user_id'] = parts[3]  # img:temp:image:user_id:timestamp
            elif content_type == 'videos' and len(parts) >= 3:
                metadata['user_id'] = parts[2]  # video:temp:user_id:timestamp
        except Exception as e:
            logger.warning(f"Impossible d'extraire l'ID utilisateur de la clé {key}: {str(e)}")

        # Stocker dans le système de fichiers
        file_path = self.file_storage.store_file(key, data, content_type)
        metadata_path = self.file_storage.store_metadata(key, metadata)

        # Vérification post-stockage
        verification_data = self.file_storage.retrieve_file(key, content_type)
        if verification_data is None:
            logger.error(
                f"Échec de vérification post-migration pour {key}. Le fichier n'est pas accessible après stockage.")
            return False

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
