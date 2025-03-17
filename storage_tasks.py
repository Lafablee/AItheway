# storage_tasks.py
import asyncio
import time
from datetime import datetime, timedelta
from flask import current_app as app


async def migrate_old_redis_content(storage_manager, age_threshold=timedelta(hours=23)):
    """Tâche d'arrière-plan pour migrer le contenu plus vieux qu'un seuil de Redis vers le disque"""
    app.logger.info("Démarrage de la tâche périodique de migration de contenu")

    while True:
        try:
            # Obtenir la liste de toutes les clés dans Redis
            keys = storage_manager.redis.keys("img:temp:image:*")
            keys.extend(storage_manager.redis.keys("audio:*"))
            keys.extend(storage_manager.redis.keys("midjourney_image:*"))

            now = datetime.now()
            migrated_count = 0

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

                    # Essayer de déterminer le type de contenu à partir des métadonnées
                    metadata = storage_manager.get_metadata(key)
                    if metadata and 'type' in metadata:
                        if metadata['type'] == 'audio':
                            content_type = 'audio'
                        elif metadata['type'] == 'video':
                            content_type = 'video'

                    # Migrer le contenu
                    success = storage_manager.migrate_to_disk(key, content_type)
                    if success:
                        migrated_count += 1

                    # Dormir brièvement pour éviter de surcharger Redis
                    await asyncio.sleep(0.01)

            app.logger.info(f"Migration terminée. {migrated_count} éléments migrés.")

            # Dormir avant la prochaine vérification
            await asyncio.sleep(3600)  # Vérifier toutes les heures

        except Exception as e:
            app.logger.error(f"Erreur dans la tâche de migration: {str(e)}")
            await asyncio.sleep(60)  # Attendre une minute avant de réessayer