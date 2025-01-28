from config import redis_client
from app import chat_manager, image_manager
import asyncio
import time


async def test_user_scenario():
    test_user_id = "test_user_456"

    # 1. Génération d'image
    print("1. Test génération d'image...")
    msg_data = {
        "text": "Générer une montagne",
        "type": "generate_image"
    }
    msg_id = chat_manager.store_message(test_user_id, msg_data)

    # 2. Vérification stockage
    print("2. Vérification du stockage...")
    stored_msg = chat_manager.get_message(msg_id)
    print(f"Message stocké: {stored_msg}")

    # 3. Chargement historique
    print("3. Test chargement historique...")
    history = chat_manager.get_user_chat_history(test_user_id)
    print(f"Historique: {history}")

    await asyncio.sleep(2)  # Simuler délai réseau

    print("Test scénario complété!")


if __name__ == "__main__":
    asyncio.run(test_user_scenario())