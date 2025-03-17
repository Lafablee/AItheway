#!/usr/bin/env python
import asyncio
import sys
import os
import logging
import signal
from logging.handlers import RotatingFileHandler
from redis import Redis
from datetime import timedelta

# Configurez le logging avant d'importer vos modules
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, 'background_worker.log'),
            maxBytes=10485760,  # 10 MB
            backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("background_worker")

# Mode test ?
TEST_MODE = os.getenv("STORAGE_TEST_MODE", "false").lower() == "true"

# Importez vos modules et tâches
from storage_manager import FileStorage, StorageManager
from storage_tasks import migrate_old_redis_content

# Configuration Redis (ajustez selon votre configuration)
redis_client = Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False
)

# Configuration du stockage
storage_path = os.getenv('STORAGE_BASE_PATH', os.path.join(os.getcwd(), 'storage'))
logger.info(f"Using storage path: {storage_path}")

try:
    # Vérifier si le répertoire existe
    if not os.path.exists(storage_path):
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"Created storage directory: {storage_path}")

    # Adapter la durée de stockage au mode test
    if TEST_MODE:
        # En mode test, utiliser une durée courte (10 minutes)
        temp_duration = timedelta(minutes=10)
        logger.info(f"Mode TEST: durée de stockage temporaire réduite à {temp_duration}")
    else:
        # En mode production, utiliser la durée normale (24 heures)
        temp_duration = timedelta(hours=24)
        logger.info(f"Mode PRODUCTION: durée de stockage temporaire de {temp_duration}")

    file_storage = FileStorage(storage_path)
    storage_manager = StorageManager(
        redis_client=redis_client,
        file_storage=file_storage,
        temp_duration=temp_duration
    )
    logger.info("Storage manager initialized successfully")
except Exception as e:
    logger.error(f"Error setting up storage manager: {str(e)}")
    sys.exit(1)

# Gestion de l'arrêt propre
shutdown_event = asyncio.Event()


def signal_handler():
    logger.info("Signal d'arrêt reçu, fermeture propre du worker...")
    shutdown_event.set()


# Runner principal pour les tâches asynchrones
async def main():
    logger.info("Démarrage du worker de tâches en arrière-plan")

    # Démarrer la tâche de migration dans un task séparé
    migration_task = asyncio.create_task(
        migrate_old_redis_content(storage_manager)
    )

    try:
        # Attendre jusqu'à ce qu'un signal d'arrêt soit reçu
        await shutdown_event.wait()
        logger.info("Arrêt propre demandé, annulation des tâches...")
    except asyncio.CancelledError:
        logger.info("Tâche principale annulée")
    finally:
        # Annuler toutes les tâches en cours
        migration_task.cancel()
        try:
            await migration_task
        except asyncio.CancelledError:
            pass

        logger.info("Toutes les tâches ont été arrêtées proprement")


if __name__ == "__main__":
    logger.info("Worker de tâches en arrière-plan démarré")

    # Configuration des gestionnaires de signaux
    loop = asyncio.get_event_loop()

    # Configurer les handlers de signaux pour arrêt propre
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"Erreur non gérée dans le worker: {str(e)}", exc_info=True)
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()

        # Attendre que toutes les tâches se terminent
        if pending:
            logger.info(f"Annulation de {len(pending)} tâches en attente")
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.close()
        logger.info("Worker arrêté proprement")