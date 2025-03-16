# gallery_manager.py corrigé - utilisez ce fichier complet pour remplacer l'existant

import json
import logging
from datetime import datetime, timedelta

# Configurer le logging standard
logger = logging.getLogger(__name__)


class GalleryManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "gallery"
        # Ne pas utiliser app.logger dans l'initialisation
        logger.info(f"GalleryManager initialized with prefix: {self.prefix}")

    def add_to_gallery(self, image_key, metadata, shared_by_user_id):
        """Version simplifiée d'ajout à la galerie"""
        try:
            # Créer une clé unique pour l'entrée de galerie
            gallery_item_id = f"{self.prefix}:item:{datetime.now().timestamp()}:{image_key}"

            # Créer des métadonnées de base
            gallery_metadata = {
                'original_key': image_key,
                'shared_by': shared_by_user_id,
                'shared_at': datetime.now().isoformat(),
                'likes': 0,
                'views': 0,
                'featured': False
            }

            # Si metadata est un dictionnaire, ajouter certaines informations
            if isinstance(metadata, dict):
                if 'prompt' in metadata:
                    gallery_metadata['prompt'] = metadata['prompt']
                if 'model' in metadata:
                    gallery_metadata['model'] = metadata['model']

            # Convertir les métadonnées en format compatible Redis
            redis_metadata = {k: str(v) for k, v in gallery_metadata.items()}

            # Stockage simple sans pipeline
            self.redis.hmset(gallery_item_id, redis_metadata)
            self.redis.expire(gallery_item_id, 60 * 60 * 24 * 90)  # 90 jours

            # Ajouter à un index simple
            self.redis.zadd(f"{self.prefix}:index", {gallery_item_id: datetime.now().timestamp()})

            return gallery_item_id

        except Exception as e:
            logger.error(f"Error adding to gallery: {str(e)}")
            return None

    def get_gallery_items(self, page=1, per_page=20, filters=None, sort_by='recent'):
        """Version simplifiée de récupération des éléments"""
        try:
            # Pagination de base
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page - 1

            # Récupérer uniquement par date (ignorer les filtres pour simplifier)
            index_key = f"{self.prefix}:index"

            # Vérifier si l'index existe
            if not self.redis.exists(index_key):
                logger.warning(f"Gallery index does not exist: {index_key}")
                # Retourner une structure vide
                return {
                    'items': [],
                    'pagination': {
                        'current_page': page,
                        'per_page': per_page,
                        'total_items': 0,
                        'has_more': False
                    }
                }

            # Récupérer les clés des éléments (plus récents d'abord)
            try:
                item_keys = self.redis.zrevrange(index_key, start_idx, end_idx)
                logger.info(f"Retrieved {len(item_keys)} keys from index")
            except Exception as e:
                logger.error(f"Error retrieving keys from index: {str(e)}")
                item_keys = []

            # Récupérer les métadonnées pour chaque clé
            items = []
            for key in item_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                try:
                    # Récupérer les métadonnées de l'élément
                    metadata = self.redis.hgetall(key_str)

                    if metadata:
                        # Convertir les bytes en strings
                        item_data = {
                            k.decode('utf-8') if isinstance(k, bytes) else k:
                                v.decode('utf-8') if isinstance(v, bytes) else v
                            for k, v in metadata.items()
                        }

                        # Ajouter l'identifiant de la galerie
                        item_data['gallery_id'] = key_str

                        # Ajouter l'URL de l'image
                        if 'original_key' in item_data:
                            item_data['image_url'] = f"/image/{item_data['original_key']}"

                        # Assigner une taille par défaut
                        item_data['size'] = ['large', 'medium', 'small'][len(items) % 3]

                        # Ajouter à la liste des éléments
                        items.append(item_data)
                except Exception as e:
                    logger.error(f"Error processing gallery item {key_str}: {str(e)}")
                    continue

            # Calculer s'il y a plus d'éléments
            try:
                total_items = self.redis.zcard(index_key)
                has_more = total_items > (start_idx + len(items))
            except Exception as e:
                logger.error(f"Error calculating pagination: {str(e)}")
                total_items = len(items)
                has_more = False

            return {
                'items': items,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': total_items,
                    'has_more': has_more
                }
            }
        except Exception as e:
            logger.error(f"Error getting gallery items: {str(e)}")
            # Retourner une structure vide en cas d'erreur
            return {
                'items': [],
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': 0,
                    'has_more': False
                }
            }

    def like_gallery_item(self, gallery_item_id, user_id):
        """Ajoute ou retire un like sur un élément de la galerie"""
        try:
            # Vérifier si l'élément existe
            if not self.redis.exists(gallery_item_id):
                return {'success': False, 'error': 'Élément non trouvé'}

            # Clé pour suivre les likes des utilisateurs
            user_like_key = f"{gallery_item_id}:likes:{user_id}"

            # Vérifier si l'utilisateur a déjà liké cet élément
            already_liked = self.redis.exists(user_like_key)

            if already_liked:
                # Retirer le like
                self.redis.delete(user_like_key)
                self.redis.hincrby(gallery_item_id, 'likes', -1)
                return {'success': True, 'action': 'unliked'}
            else:
                # Ajouter le like
                self.redis.setex(user_like_key, 60 * 60 * 24 * 365, 1)  # TTL d'un an
                self.redis.hincrby(gallery_item_id, 'likes', 1)
                return {'success': True, 'action': 'liked'}

        except Exception as e:
            logger.error(f"Error liking gallery item: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_gallery_item(self, gallery_item_id):
        """Récupère les détails d'un élément spécifique de la galerie"""
        try:
            if not self.redis.exists(gallery_item_id):
                return None

            metadata = self.redis.hgetall(gallery_item_id)

            # Décoder les valeurs bytes en strings
            metadata_dict = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                    v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in metadata.items()
            }

            # Transformer la clé en URL
            original_key = metadata_dict.get('original_key')
            if original_key:
                metadata_dict['image_url'] = f"/image/{original_key}"

            # Incrémenter le compteur de vues
            self.redis.hincrby(gallery_item_id, 'views', 1)

            return metadata_dict

        except Exception as e:
            logger.error(f"Error retrieving gallery item: {str(e)}")
            return None