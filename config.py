
import redis
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

redis_client = redis.Redis(

    host=os.getenv('REDIS_HOST', 'localhost'),

    port=int(os.getenv('REDIS_PORT', 6379)),

    db=int(os.getenv('REDIS_DB', 0)),

    decode_responses=False
)

TEMP_STORAGE_DURATION = 24 * 60 * 60  # 24 heures en secondes
PERMANENT_STORAGE_DURATION = 30 * 24 * 60 * 60  # 30 jours en secondes