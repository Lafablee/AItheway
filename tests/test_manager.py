# test_managers.py
from config import redis_client
from app import ImageManager, ChatManager
from datetime import datetime
import io

# Initialisation des managers pour les tests
image_manager = ImageManager(redis_client)
chat_manager = ChatManager(redis_client)


def test_chat_manager():
    test_user_id = "test_user_123"

    try:
        # Test stockage message
        message_data = {
            "text": "Test message",
            "type": "text"
        }

        msg_id = chat_manager.store_message(test_user_id, message_data)
        print(f"Message ID created: {msg_id}")

        # Test récupération
        stored_message = chat_manager.get_message(msg_id)
        print(f"Stored message retrieved: {stored_message}")

        # Test historique
        history = chat_manager.get_user_chat_history(test_user_id, page=1)
        print(f"User history: {history}")

        print("Chat manager tests completed successfully")

    except Exception as e:
        print(f"Chat manager test failed: {str(e)}")


def test_image_manager():
    test_user_id = "test_user_123"

    try:
        # Créer des données d'image de test
        test_image_data = b"test image data"
        test_metadata = {
            "type": "test",
            "filename": "test.png"
        }

        # Test stockage
        image_key = image_manager.store_temp_image(
            test_user_id,
            test_image_data,
            test_metadata
        )
        print(f"Image stored with key: {image_key}")

        # Test récupération
        stored_image = image_manager.get_image(image_key)
        print(f"Image retrieved successfully: {stored_image == test_image_data}")

        # Test historique
        history = image_manager.get_user_history(test_user_id)
        print(f"Image history: {history}")

        print("Image manager tests completed successfully")

    except Exception as e:
        print(f"Image manager test failed: {str(e)}")


if __name__ == "__main__":
    print("Testing Chat Manager...")
    test_chat_manager()
    print("\nTesting Image Manager...")
    test_image_manager()