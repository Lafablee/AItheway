#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de récupération des médias manquants dans la bibliothèque.
Ce script examine la bibliothèque pour identifier les vidéos et audios qui
n'ont pas été téléchargés et tente de les récupérer depuis leurs URLs stockées.
"""

import os
import sys
import json
import time
import logging
import argparse
import redis
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('media_recovery.log')
    ]
)
logger = logging.getLogger("media_recovery")

# Configuration Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Configuration du stockage
STORAGE_BASE_PATH = os.getenv("STORAGE_BASE_PATH", os.path.join(os.getcwd(), 'storage'))


def connect_redis():
    """Établit une connexion à Redis"""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False  # Important pour les données binaires
        )
        # Tester la connexion
        client.ping()
        logger.info(f"Redis connection established to {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        sys.exit(1)


def decode_metadata(data):
    """Décode les métadonnées Redis en structure Python"""
    if not data:
        return {}

    result = {}
    for k, v in data.items():
        key = k.decode('utf-8') if isinstance(k, bytes) else k
        # Essayer de décoder en string, sinon garder tel quel
        if isinstance(v, bytes):
            try:
                value = v.decode('utf-8')
            except UnicodeDecodeError:
                value = v
        else:
            value = v
        result[key] = value

    return result


def find_all_media(redis_client, media_type='video'):
    """
    Trouve tous les médias d'un type spécifique.

    Args:
        redis_client: Client Redis
        media_type: 'video', 'audio' ou 'image'

    Returns:
        Liste de clés de médias avec leurs métadonnées
    """
    # Définir le pattern selon le type de média
    if media_type == 'video':
        pattern = "video:*"
    elif media_type == 'audio':
        pattern = "audio:*"
    elif media_type == 'image':
        pattern = "img:temp:image:*"
    else:
        raise ValueError(f"Type de média non supporté: {media_type}")

    # Récupérer toutes les clés correspondant au pattern
    all_keys = redis_client.keys(pattern)

    # Filtrer pour exclure les clés de métadonnées
    media_keys = [k for k in all_keys if not k.endswith(b':meta')]

    logger.info(f"Found {len(media_keys)} {media_type} keys in Redis")

    # Récupérer les métadonnées pour chaque clé
    media_items = []
    for key in media_keys:
        # Convertir la clé en string
        key_str = key.decode('utf-8') if isinstance(key, bytes) else key

        # Récupérer les métadonnées
        meta_key = f"{key_str}:meta"
        metadata = redis_client.hgetall(meta_key)

        if metadata:
            # Décoder les métadonnées
            decoded_metadata = decode_metadata(metadata)
            decoded_metadata['key'] = key_str
            media_items.append(decoded_metadata)

    logger.info(f"Retrieved metadata for {len(media_items)} {media_type}s")
    return media_items


def find_missing_videos(redis_client, downloaded_only=False):
    """
    Identifie les vidéos qui n'ont pas de fichier local ou dont le téléchargement a échoué

    Args:
        redis_client: Client Redis
        downloaded_only: Si True, ne considère que les vidéos déjà marquées comme téléchargées

    Returns:
        Liste de vidéos manquantes
    """
    videos = find_all_media(redis_client, 'video')

    missing_videos = []
    for video in videos:
        video_key = video.get('key')
        if not video_key:
            continue

        # Vérifier si le fichier existe sur le disque
        file_path = get_file_path(video_key, 'videos')
        file_exists = os.path.exists(file_path) if file_path else False

        # Vérifier le statut du fichier selon les métadonnées
        file_stored = video.get('file_stored') == 'true' or video.get('file_stored') == True
        download_status = video.get('download_status', '')

        if (file_stored and not file_exists) or (downloaded_only and not file_exists):
            # Le fichier est marqué comme téléchargé mais n'existe pas
            missing_videos.append({
                'key': video_key,
                'status': video.get('status', 'unknown'),
                'download_status': download_status,
                'download_url': video.get('download_url', ''),
                'task_id': video.get('task_id', ''),
                'timestamp': video.get('timestamp', '')
            })
        elif not downloaded_only and not file_exists and video.get('download_url'):
            # Le fichier n'existe pas mais une URL est disponible
            missing_videos.append({
                'key': video_key,
                'status': video.get('status', 'unknown'),
                'download_status': download_status,
                'download_url': video.get('download_url', ''),
                'task_id': video.get('task_id', ''),
                'timestamp': video.get('timestamp', '')
            })

    logger.info(f"Found {len(missing_videos)} missing videos")
    return missing_videos


def find_missing_audio(redis_client):
    """
    Identifie les fichiers audio qui n'ont pas de fichier local

    Args:
        redis_client: Client Redis

    Returns:
        Liste d'audios manquants
    """
    audios = find_all_media(redis_client, 'audio')

    missing_audios = []
    for audio in audios:
        audio_key = audio.get('key')
        if not audio_key:
            continue

        # Vérifier si le fichier existe sur le disque
        file_path = get_file_path(audio_key, 'audio')
        file_exists = os.path.exists(file_path) if file_path else False

        if not file_exists:
            missing_audios.append({
                'key': audio_key,
                'text': audio.get('text', ''),
                'timestamp': audio.get('timestamp', ''),
                'voice': audio.get('voice', '')
            })

    logger.info(f"Found {len(missing_audios)} missing audio files")
    return missing_audios


def get_file_path(key, content_type):
    """
    Calcule le chemin du fichier sur disque pour une clé donnée

    Args:
        key: Clé du média
        content_type: Type de contenu ('videos', 'audio', 'images')

    Returns:
        Chemin complet du fichier ou None si la clé est invalide
    """
    import hashlib

    # Calculer le hash de la clé pour le système de fichiers
    if isinstance(key, bytes):
        key_str = key.decode('utf-8')
    else:
        key_str = key

    key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()

    # Créer le chemin selon la structure en arborescence
    dir1, dir2 = key_hash[:2], key_hash[2:4]
    path = os.path.join(STORAGE_BASE_PATH, content_type, dir1, dir2, key_hash)

    return path


def download_file(url, timeout=60):
    """
    Télécharge un fichier depuis une URL

    Args:
        url: URL du fichier
        timeout: Timeout en secondes

    Returns:
        Contenu du fichier ou None en cas d'échec
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)

        if response.status_code == 403:
            logger.warning(f"URL expired (403): {url}")
            return None

        response.raise_for_status()

        # Lire le contenu
        return response.content

    except Exception as e:
        logger.error(f"Error downloading from {url}: {str(e)}")
        return None


def refresh_minimax_url(task_id):
    """
    Tente de rafraîchir une URL MiniMax expirée

    Args:
        task_id: ID de la tâche MiniMax

    Returns:
        Nouvelle URL ou None en cas d'échec
    """
    try:
        # Importer le gestionnaire de AI
        from ai_models import create_ai_manager
        ai_manager = create_ai_manager()

        minimax_generator = ai_manager.generators.get("minimax-video")

        if not minimax_generator:
            logger.error("MiniMax video generator not available")
            return None

        # Vérifier le statut pour obtenir le file_id
        status_response = minimax_generator.generator.check_generation_status(task_id)

        if status_response.get('success') and status_response.get('file_id'):
            file_id = status_response.get('file_id')

            # Obtenir une nouvelle URL
            new_url = minimax_generator.generator.get_download_url(file_id)

            if new_url:
                logger.info(f"URL refreshed successfully for task {task_id}: {new_url}")
                return new_url

        logger.warning(f"Failed to refresh URL for task {task_id}")
        return None

    except Exception as e:
        logger.error(f"Error refreshing URL for task {task_id}: {str(e)}")
        return None


def store_file(key, data, content_type):
    """
    Stocke un fichier sur le disque

    Args:
        key: Clé du fichier
        data: Contenu binaire du fichier
        content_type: Type de contenu ('videos', 'audio', 'images')

    Returns:
        Booléen indiquant le succès de l'opération
    """
    try:
        file_path = get_file_path(key, content_type)

        if not file_path:
            logger.error(f"Invalid key: {key}")
            return False

        # Créer les répertoires si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Écrire le fichier
        with open(file_path, 'wb') as f:
            f.write(data)

        logger.info(f"File stored successfully at {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error storing file {key}: {str(e)}")
        return False


def update_metadata(redis_client, key, updates):
    """
    Met à jour les métadonnées d'un fichier

    Args:
        redis_client: Client Redis
        key: Clé du média
        updates: Dictionnaire des mises à jour

    Returns:
        Booléen indiquant le succès de l'opération
    """
    try:
        meta_key = f"{key}:meta"

        # Convertir les valeurs en bytes pour Redis
        redis_updates = {}
        for k, v in updates.items():
            if isinstance(v, str):
                redis_updates[k] = v.encode('utf-8')
            elif isinstance(v, bool):
                redis_updates[k] = str(v).lower().encode('utf-8')
            elif v is None:
                redis_updates[k] = b''
            else:
                redis_updates[k] = v

        # Mettre à jour les métadonnées
        redis_client.hmset(meta_key, redis_updates)

        return True

    except Exception as e:
        logger.error(f"Error updating metadata for {key}: {str(e)}")
        return False


def recover_videos(redis_client, videos, retry_refresh=True, max_retries=3):
    """
    Tente de récupérer les vidéos manquantes

    Args:
        redis_client: Client Redis
        videos: Liste des vidéos à récupérer
        retry_refresh: Si True, tente de rafraîchir les URLs expirées
        max_retries: Nombre maximal de tentatives par vidéo

    Returns:
        Tuples (réussites, échecs)
    """
    success_count = 0
    failed_count = 0

    for video in tqdm(videos, desc="Récupération des vidéos"):
        key = video.get('key')
        download_url = video.get('download_url')
        task_id = video.get('task_id')

        if not download_url:
            logger.warning(f"No download URL for video {key}")
            failed_count += 1
            continue

        # Tentatatives de téléchargement
        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries} for video {key}")

            # Télécharger la vidéo
            video_data = download_file(download_url)

            if video_data:
                # Téléchargement réussi, stocker la vidéo
                if store_file(key, video_data, 'videos'):
                    # Mettre à jour les métadonnées
                    update_metadata(redis_client, key, {
                        'file_stored': True,
                        'download_status': 'complete',
                        'file_size': len(video_data),
                        'recovery_date': datetime.now().isoformat()
                    })
                    success_count += 1
                    break

            elif retry_refresh and task_id and attempt < max_retries - 1:
                # Échec du téléchargement, essayer de rafraîchir l'URL
                logger.info(f"Trying to refresh URL for video {key} (task {task_id})")
                new_url = refresh_minimax_url(task_id)

                if new_url:
                    download_url = new_url
                    # Mettre à jour l'URL dans les métadonnées
                    update_metadata(redis_client, key, {
                        'download_url': new_url
                    })
                    # Petite pause avant de réessayer
                    time.sleep(1)
                else:
                    # Échec du rafraîchissement, passer à la prochaine vidéo
                    break
            else:
                # Pas de rafraîchissement ou dernier essai, passer à la prochaine vidéo
                break
        else:
            # Toutes les tentatives ont échoué
            update_metadata(redis_client, key, {
                'download_status': 'failed',
                'recovery_attempts': max_retries,
                'last_attempt': datetime.now().isoformat()
            })
            failed_count += 1

    return success_count, failed_count


def main():
    """Fonction principale du script de récupération"""
    parser = argparse.ArgumentParser(description='Script de récupération des médias manquants')
    parser.add_argument('--type', choices=['video', 'audio', 'all'], default='all',
                        help='Type de média à récupérer')
    parser.add_argument('--downloaded-only', action='store_true',
                        help='Ne récupérer que les médias déjà marqués comme téléchargés')
    parser.add_argument('--no-refresh', action='store_true',
                        help='Ne pas essayer de rafraîchir les URLs expirées')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Nombre maximal de tentatives par média')

    args = parser.parse_args()

    # Connecter à Redis
    redis_client = connect_redis()

    # Récupérer les médias manquants
    if args.type in ['video', 'all']:
        missing_videos = find_missing_videos(redis_client, args.downloaded_only)

        if missing_videos:
            logger.info(f"Attempting to recover {len(missing_videos)} videos")
            success, failed = recover_videos(
                redis_client,
                missing_videos,
                retry_refresh=not args.no_refresh,
                max_retries=args.max_retries
            )
            logger.info(f"Video recovery completed: {success} successes, {failed} failures")
        else:
            logger.info("No missing videos to recover")

    # TODO: Implémenter la récupération des fichiers audio si nécessaire

    logger.info("Media recovery process completed")


if __name__ == "__main__":
    main()