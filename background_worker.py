#!/usr/bin/env python
import asyncio
import sys
import os
import logging
from logging.handlers import RotatingFileHandler

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

# Importez vos modules et tâches
from redis import Redis
from storage_manager import FileStorage, StorageManager
from storage_tasks import migrate_old_redis_content
from datetime import timedelta

# Configuration Redis (ajustez selon votre configuration)
redis_client = Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False
)

# Configuration du stockage
storage_path = os.getenv('STORAGE_BASE_PATH', os.path.join(os.getcwd(), 'storage'))
file_storage = FileStorage(storage_path)
storage_manager = StorageManager(
    redis_client=redis_client,
    file_storage=file_storage,
    temp_duration=timedelta(hours=24)
)


# Runner principal pour les tâches asynchrones
async def main():
    logger.info("Démarrage du worker de tâches en arrière-plan")

    # Pour ajouter d'autres tâches, ajoutez-les à cette liste
    tasks = [
        migrate_old_redis_content(storage_manager)
    ]

    # Si vous voulez ajouter des tâches Midjourney
    # from tasks import check_pending_midjourney_tasks
    # tasks.append(check_pending_midjourney_tasks())

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logger.info("Worker de tâches en arrière-plan démarré")
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Arrêt du worker demandé, fermeture propre...")
    except Exception as e:
        logger.error(f"Erreur non gérée dans le worker: {str(e)}", exc_info=True)
    finally:
        loop.close()
        logger.info("Worker arrêté")