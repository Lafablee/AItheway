# storage_tasks.py
import asyncio
import logging
import time
from datetime import datetime, timedelta
from venv import logger

from flask import current_app as app

logger = logging.getLogger("background_worker")

async def migrate_old_redis_content(storage_manager, age_threshold=timedelta(hours=23)):
    """Tâche d'arrière-plan pour migrer le contenu plus vieux qu'un seuil de Redis vers le disque"""
    logger.info("Démarrage de la tâche périodique de migration de contenu")

    while True:
        try:
            # Obtenir la liste de toutes les clés dans Redis
            keys = storage_manager.redis.keys("img:temp:image:*")
            keys.extend(storage_manager.redis.keys("midjourney_image:*"))

            keys.extend(storage_manager.redis.keys("audio:*"))

            logger.info(f"Trouvé {len(keys)} clés à analyser pour migration potentielle")
            now = datetime.now()
            migrated_count = 0
            migrated_types = {"images": 0, "audio": 0, "video": 0}

            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')

                # Ignorer les clés de métadonnées
                if key.endswith(":meta"):
                    continue

                # Vérifier le TTL pour estimer l'âge
                ttl = storage_manager.redis.ttl(key)

                # Si le TTL est moins que le seuil, migrer vers le disque
                original_ttl = storage_manager.temp_duration.total_seconds()
                if ttl < original_ttl - age_threshold.total_seconds():
                    content_type = 'images'  # Par défaut

                    # Détection automatique basée sur le préfixe de la clé
                    if key.startswith("audio."):
                        content_type = 'audio'
                    elif key.startswith("video."):
                        content_type = 'video'

                    # Essayer de déterminer le type de contenu à partir des métadonnées
                    metadata = storage_manager.get_metadata(key)
                    if metadata and 'type' in metadata:
                        meta_type = metadata['type']
                        if meta_type == 'audio':
                            content_type = 'audio'
                        elif meta_type == 'video':
                            content_type = 'video'

                    # Migrer le contenu
                    logger.info(f"Migration de {key} (type: {content_type}) vers le stockage disque")
                    success = storage_manager.migrate_to_disk(key, content_type)

                    if success:
                        migrated_count += 1
                        migrated_types[content_type] += 1
                        logger.debug(f"Migration résussie pour {key}")
                    else:
                        logger.warning(f"Echec de la migration pour {key}")

                    # Dormir brièvement pour éviter de surcharger Redis
                    await asyncio.sleep(0.01)

            if migrated_count > 0:
                logger.info(f"Migration terminée. {migrated_count} éléments migrés:"
                            f"{migrated_types['images']} images, "
                            f"{migrated_types['audio']} audio, "
                            f"{migrated_types['video']} vidéo.")
            else:
                logger.info(f"Aucun élément à migrer dans cette exécution.")

            # Dormir avant la prochaine vérification
            await asyncio.sleep(3600)  # Vérifier toutes les heures

        except Exception as e:
            logger.error(f"Erreur dans la tâche de migration: {str(e)}")
            await asyncio.sleep(60)  # Attendre une minute avant de réessayer