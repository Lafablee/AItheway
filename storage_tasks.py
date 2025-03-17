# storage_tasks.py
import asyncio
import logging
import os
import sys
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

    # Logique de détermination de l'âge à partir du TTL
    def should_migrate(key, ttl):
        try:
            # Obtenir la durée originale de stockage
            original_ttl = int(storage_manager.temp_duration.total_seconds())

            # Calculer le seuil en secondes
            threshold_seconds = age_threshold.total_seconds()

            # Calculer le temps écoulé depuis le stockage
            elapsed_time = original_ttl - ttl

            # Migrer si le temps écoulé est supérieur au seuil
            should_migrate = elapsed_time >= threshold_seconds

            # Log de diagnostic détaillé
            if TEST_MODE:
                logger.info(f"Diagnostic de migration - Clé: {key}")
                logger.info(f"  → TTL actuel: {ttl}s")
                logger.info(f"  → TTL original: {original_ttl}s")
                logger.info(f"  → Temps écoulé: {elapsed_time}s")
                logger.info(f"  → Seuil de migration: {threshold_seconds}s")
                logger.info(f"  → Décision de migration: {should_migrate}")

            return should_migrate
        except Exception as e:
            logger.error(f"Erreur dans should_migrate pour {key}: {str(e)}")
            return False

    logger.info(f"Démarrage de la tâche périodique de migration de contenu (seuil: {age_threshold})")

    # Boucle de vérification continuelle
    while True:
        try:
            start_time = datetime.now()
            logger.info(f"[{start_time.strftime('%H:%M:%S')}] Début de vérification des clés pour migration...")

            # Obtenir la liste de toutes les clés dans Redis
            all_keys = []

            # Images
            image_keys = storage_manager.redis.keys("img:temp:image:*")
            all_keys.extend(image_keys)

            # Images Midjourney
            mj_keys = storage_manager.redis.keys("midjourney_image:*")
            all_keys.extend(mj_keys)

            # Audio
            audio_keys = storage_manager.redis.keys("audio:*")
            all_keys.extend(audio_keys)

            # Filtrer les clés de métadonnées
            filtered_keys = []
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if not key_str.endswith(":meta"):
                    filtered_keys.append(key_str)

            logger.info(f"Trouvé {len(filtered_keys)} clés à analyser pour migration potentielle")

            # Variable pour suivre les statistiques de migration
            migrated_count = 0
            migrated_types = {"images": 0, "audio": 0, "video": 0}

            # Si nous sommes en mode test, afficher des infos pour toutes les clés
            if TEST_MODE and filtered_keys:
                logger.info("=== Détails des clés trouvées ===")
                for i, key in enumerate(filtered_keys[:10]):  # Limiter à 10 pour éviter des logs trop longs
                    ttl = storage_manager.redis.ttl(key)
                    logger.info(f"Clé #{i + 1}: {key} (TTL: {ttl}s)")
                if len(filtered_keys) > 10:
                    logger.info(f"...et {len(filtered_keys) - 10} autres clés")

            # Parcourir les clés pour migration
            for key in filtered_keys:
                # Vérifier le TTL
                ttl = storage_manager.redis.ttl(key)

                # Déterminer si la clé doit être migrée
                if should_migrate(key, ttl):
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
                        logger.info(f"Migration réussie pour {key}")
                    else:
                        logger.warning(f"Échec de la migration pour {key}")

                    # Dormir brièvement pour éviter de surcharger Redis
                    await asyncio.sleep(0.01)

            # Bilan de la migration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if migrated_count > 0:
                logger.info(f"Migration terminée en {duration:.2f}s. {migrated_count} éléments migrés: "
                            f"{migrated_types['images']} images, "
                            f"{migrated_types['audio']} audio, "
                            f"{migrated_types['video']} vidéo.")
            else:
                logger.info(f"Aucun élément à migrer dans cette exécution (durée: {duration:.2f}s).")

            # Dormir avant la prochaine vérification - durée basée sur le mode
            check_interval = TEST_CHECK_SECONDS if TEST_MODE else 3600  # 30 secondes en test, 1 heure en prod
            next_check = datetime.now() + timedelta(seconds=check_interval)
            logger.info(
                f"Prochaine vérification prévue à {next_check.strftime('%H:%M:%S')} (dans {check_interval} secondes)")
            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Erreur dans la tâche de migration: {str(e)}", exc_info=True)
            # Pause plus courte en cas d'erreur en mode test
            error_wait = 5 if TEST_MODE else 60
            logger.info(f"Nouvelle tentative dans {error_wait} secondes...")
            await asyncio.sleep(error_wait)