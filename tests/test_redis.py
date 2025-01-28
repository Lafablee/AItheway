# test_redis.py
import os
import sys

# Ajoute le répertoire parent au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import redis_client
import redis


def test_redis_connection():
    try:
        response = redis_client.ping()
        print("Redis connection test:", "SUCCESS" if response else "FAILED")

        # Test basique de stockage/récupération
        redis_client.set('test_key', 'test_value')
        test_value = redis_client.get('test_key')
        print("Redis read/write test:", "SUCCESS" if test_value == b'test_value' else "FAILED")

        # Nettoyage
        redis_client.delete('test_key')

    except redis.ConnectionError as e:
        print("Redis connection failed:", str(e))
    except Exception as e:
        print("Test failed with error:", str(e))


if __name__ == "__main__":
    test_redis_connection()