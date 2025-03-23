# storage_tasks.py
import asyncio
import logging
import os
from datetime import datetime, timedelta

# Obtenir le logger configuré dans background_worker.py
logger = logging.getLogger("background_worker")

# Lire les paramètres d'environnement pour le mode test
TEST_MODE = os.getenv("STORAGE_TEST_MODE", "false").lower() == "true"
TEST_AGE_MINUTES = int(os.getenv("STORAGE_TEST_AGE_MINUTES", "5"))
TEST_CHECK_SECONDS = int(os.getenv("STORAGE_TEST_CHECK_SECONDS", "30"))

logger.info(f"Mode test activé: {TEST_MODE}")
if TEST_MODE:
    logger.info(
        f"Paramètres de test: Âge de migration: {TEST_AGE_MINUTES} minutes, Intervalle de vérification: {TEST_CHECK_SECONDS} secondes")


async def migrate_old_redis_content(storage_manager, age_threshold=None):
    """
    Tâche d'arrière-plan pour migrer le contenu plus vieux qu'un seuil de Redis vers le disque

    En mode test:
    - age_threshold est réduit à quelques minutes (défini par STORAGE_TEST_AGE_MINUTES)
    - L'intervalle de vérification est réduit à quelques secondes (défini par STORAGE_TEST_CHECK_SECONDS)
    """
    # Déterminer le seuil d'âge selon le mode
    if age_threshold is None:
        if TEST_MODE:
            age_threshold = timedelta(minutes=TEST_AGE_MINUTES)
            logger.info(
                f"Mode TEST activé: Migration après {TEST_AGE_MINUTES} minutes, vérification toutes les {TEST_CHECK_SECONDS} secondes")
        else:
            age_threshold = timedelta(hours=23)
            logger.info(f"Mode PRODUCTION: Migration après 23 heures, vérification toutes les heures")

    logger.info(f"Démarrage de la tâche périodique de migration de contenu (seuil: {age_threshold})")

    while True:
        try:
            # Obtenir la liste de toutes les clés dans Redis
            # Images
            keys = storage_manager.redis.keys("img:temp:image:*")
            keys.extend(storage_manager.redis.keys("midjourney_image:*"))

            # Audio
            keys.extend(storage_manager.redis.keys("audio:*"))

            # Video
            keys.extend(storage_manager.redis.keys("video:*"))

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
                    # Déterminer le type de contenu
                    content_type = 'images'  # Par défaut

                    # Détection automatique basée sur le préfixe de la clé
                    if key.startswith("audio:"):
                        content_type = 'audio'
                    elif key.startswith("video:"):
                        content_type = 'video'

                    # Vérification supplémentaire dans les métadonnées
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
                        logger.debug(f"Migration réussie pour {key}")

                        # Vérification post-migration
                        verification_data = storage_manager.file_storage.retrieve_file(key, content_type)
                        if verification_data:
                            logger.debug(f"Vérification post-migration réussie pour {key}")
                        else:
                            logger.warning(f"Vérification post-migration échouée pour {key}")
                    else:
                        logger.warning(f"Échec de la migration pour {key}")

                    # Dormir brièvement pour éviter de surcharger Redis
                    await asyncio.sleep(0.01)

            if migrated_count > 0:
                logger.info(f"Migration terminée. {migrated_count} éléments migrés: "
                            f"{migrated_types['images']} images, "
                            f"{migrated_types['audio']} audio, "
                            f"{migrated_types['video']} vidéo.")
            else:
                logger.info("Aucun élément à migrer dans cette exécution.")

            # Dormir avant la prochaine vérification - durée basée sur le mode
            check_interval = TEST_CHECK_SECONDS if TEST_MODE else 3600  # 30 secondes en test, 1 heure en prod
            logger.debug(f"Pause de {check_interval} secondes avant la prochaine vérification")
            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Erreur dans la tâche de migration: {str(e)}", exc_info=True)
            # Pause plus courte en cas d'erreur en mode test
            error_wait = 5 if TEST_MODE else 60
            await asyncio.sleep(error_wait)