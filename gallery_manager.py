import json
from datetime import datetime, timedelta
import redis
from flask import current_app as app


class GalleryManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "gallery"

    def add_to_gallery(self, image_key, metadata, shared_by_user_id):
        """
        Ajoute une image à la galerie publique

        Args:
            image_key (str): Clé Redis de l'image
            metadata (dict): Métadonnées de l'image
            shared_by_user_id (str): ID de l'utilisateur qui partage
        """
        try:
            # Créer une clé unique pour l'entrée de galerie
            gallery_item_id = f"{self.prefix}:item:{datetime.now().timestamp()}:{image_key}"

            # Ajouter les métadonnées supplémentaires pour la galerie
            gallery_metadata = {
                'original_key': image_key,
                'shared_by': shared_by_user_id,
                'shared_at': datetime.now().isoformat(),
                'likes': 0,
                'views': 0,
                'featured': False,
                'original_metadata': json.dumps(metadata)
            }

            # Utiliser un pipeline pour les opérations atomiques
            pipe = self.redis.pipeline()

            # Stocker les métadonnées de la galerie
            pipe.hmset(gallery_item_id, gallery_metadata)

            # Définir une durée de vie (par exemple, 90 jours)
            pipe.expire(gallery_item_id, 60 * 60 * 24 * 90)  # 90 jours

            # Ajouter à l'index global de la galerie (trié par date de partage)
            pipe.zadd(f"{self.prefix}:index", {gallery_item_id: datetime.now().timestamp()})

            # Ajouter aux indices de filtrage si les métadonnées contiennent des informations pertinentes
            if isinstance(metadata, dict):
                # Exemple: indexer par modèle AI
                if 'model' in metadata:
                    model = metadata.get('model')
                    pipe.sadd(f"{self.prefix}:filter:model:{model}", gallery_item_id)

                # Ajouter d'autres indices selon les besoins (environnement, mouvement, etc.)
                # Ces valeurs devraient être extraites du prompt ou des paramètres

                # Exemple avec des catégories extraites du prompt
                prompt = metadata.get('prompt', '')

                # Extraction simplifiée - à améliorer avec NLP ou règles plus sophistiquées
                environments = ['forêt', 'urbain', 'plage', 'montagne', 'désert', 'mer']
                movements = ['marche', 'course', 'vol', 'conduite', 'nage']

                for env in environments:
                    if env.lower() in prompt.lower():
                        pipe.sadd(f"{self.prefix}:filter:environment:{env}", gallery_item_id)

                for mov in movements:
                    if mov.lower() in prompt.lower():
                        pipe.sadd(f"{self.prefix}:filter:movement:{mov}", gallery_item_id)

            # Exécuter toutes les opérations
            pipe.execute()

            app.logger.info(f"Image {image_key} ajoutée à la galerie avec succès")
            return gallery_item_id

        except Exception as e:
            app.logger.error(f"Erreur lors de l'ajout à la galerie: {str(e)}")
            return None

    def get_gallery_items(self, page=1, per_page=20, filters=None, sort_by='recent'):
        """
        Récupère les éléments de la galerie avec pagination et filtrage

        Args:
            page (int): Numéro de page
            per_page (int): Nombre d'éléments par page
            filters (dict): Filtres à appliquer (model, environment, movement, etc.)
            sort_by (str): Critère de tri ('recent', 'popular', 'views')

        Returns:
            dict: Données de la galerie paginées et filtrées
        """
        try:
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page - 1

            # Gérer le filtrage
            item_keys = []

            if filters and any(filters.values()):
                # Récupérer les clés correspondant à chaque filtre
                filter_keys = []

                # Vérifier que filters est bien un dictionnaire
                if not isinstance(filters, dict):
                    app.logger.error(f"filters n'est pas un dictionnaire: {type(filters)}")
                    filters = {}

                # Vérifier que items() est bien disponible
                try:
                    filter_items = filters.items()
                except AttributeError:
                    app.logger.error(f"filters n'a pas de méthode items(): {type(filters)}")
                    filter_items = []

                for filter_type, filter_value in filter_items:
                    if filter_value and isinstance(filter_value, str) and filter_value.lower() != 'tous':
                        filter_set = f"{self.prefix}:filter:{filter_type}:{filter_value}"
                        try:
                            members = self.redis.smembers(filter_set)
                            if not callable(members):  # Vérifier que ce n'est pas une fonction
                                filter_keys.append(set(members))
                            else:
                                app.logger.error(f"smembers a retourné une fonction au lieu d'un ensemble")
                        except Exception as e:
                            app.logger.error(f"Erreur lors de la récupération des membres du filtre {filter_set}: {e}")

                # Intersection des ensembles pour obtenir les éléments correspondant à TOUS les filtres
                if filter_keys:
                    try:
                        if len(filter_keys) > 0:
                            intersection = set.intersection(*filter_keys)
                            item_keys = list(intersection)
                        else:
                            item_keys = []
                    except Exception as e:
                        app.logger.error(f"Erreur lors de l'intersection des filtres: {e}")
                        item_keys = []

                # Si aucun résultat après filtrage
                if not item_keys:
                    return {
                        'items': [],
                        'pagination': {
                            'current_page': page,
                            'per_page': per_page,
                            'total_items': 0,
                            'has_more': False
                        }
                    }
            else:
                # Sans filtre, utiliser l'index principal trié
                if sort_by == 'recent':
                    # Tri par date (plus récent en premier)
                    item_keys = self.redis.zrevrange(f"{self.prefix}:index", start_idx, end_idx)
                elif sort_by == 'popular':
                    # Trier par nombre de likes
                    # Nécessite un index séparé ou une requête plus complexe
                    # Pour simplifier, on utilise l'ordre par date pour l'instant
                    item_keys = self.redis.zrevrange(f"{self.prefix}:index", start_idx, end_idx)
                elif sort_by == 'views':
                    # Trier par nombre de vues
                    # Même approche simplifiée
                    item_keys = self.redis.zrevrange(f"{self.prefix}:index", start_idx, end_idx)

            # Récupérer les métadonnées pour chaque clé
            gallery_items = []

            if not item_keys and filters and any(filters.values()):
                # Si filtrage actif mais pas d'éléments, retourner liste vide
                return {
                    'items': [],
                    'pagination': {
                        'current_page': page,
                        'per_page': per_page,
                        'total_items': 0,
                        'has_more': False
                    }
                }

            # Si aucun élément avec le filtrage standard, récupérer tous les éléments
            if not item_keys:
                total_items = self.redis.zcard(f"{self.prefix}:index")
                item_keys = self.redis.zrevrange(f"{self.prefix}:index", start_idx, end_idx)
            else:
                # Si nous avons filtré, nous devons calculer le total
                total_items = len(item_keys)
                # Et limiter les résultats à la page demandée
                item_keys = item_keys[start_idx:end_idx + 1]

            # Utiliser un pipeline pour récupérer efficacement toutes les métadonnées
            pipe = self.redis.pipeline()

            for key in item_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                pipe.hgetall(key)

            results = pipe.execute()

            for i, metadata in enumerate(results):
                if metadata:
                    key = item_keys[i].decode('utf-8') if isinstance(item_keys[i], bytes) else item_keys[i]

                    # Décoder les valeurs bytes en strings
                    metadata_dict = {
                        k.decode('utf-8') if isinstance(k, bytes) else k:
                            v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in metadata.items()
                    }

                    # Charger les métadonnées originales
                    if 'original_metadata' in metadata_dict:
                        try:
                            original_metadata = json.loads(metadata_dict['original_metadata'])

                            # Fusionner avec les métadonnées de galerie
                            item_data = {**metadata_dict, **original_metadata}

                            # Transformer la clé en URL
                            original_key = metadata_dict.get('original_key')
                            if original_key:
                                item_data['image_url'] = f"/image/{original_key}"

                            # Ajouter un identifiant unique pour la galerie
                            item_data['gallery_id'] = key

                            # Déterminer la taille pour l'affichage
                            # Logique simple basée sur le modèle ou d'autres attributs
                            if original_metadata.get('model') == 'midjourney':
                                item_data['size'] = 'large'
                            elif 'featured' in metadata_dict and metadata_dict['featured'] == 'True':
                                item_data['size'] = 'large'
                            else:
                                # Alternance des tailles pour une galerie visuellement intéressante
                                item_data['size'] = ['medium', 'small'][i % 2]

                            gallery_items.append(item_data)

                        except json.JSONDecodeError:
                            app.logger.error(
                                f"Erreur lors du décodage des métadonnées originales: {metadata_dict.get('original_metadata')}")
                            continue

            # Incrémenter le compteur de vues pour les éléments consultés
            if gallery_items and sort_by != 'views':  # Éviter de modifier le compteur lors du tri par vues
                view_pipe = self.redis.pipeline()
                for item in gallery_items:
                    if 'gallery_id' in item:
                        view_pipe.hincrby(item['gallery_id'], 'views', 1)
                view_pipe.execute()

            # Vérifier s'il y a plus d'éléments (pour pagination)
            has_more = total_items > (start_idx + len(gallery_items))

            return {
                'items': gallery_items,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_items': total_items,
                    'has_more': has_more
                }
            }

        except Exception as e:
            app.logger.error(f"Erreur lors de la récupération des éléments de la galerie: {str(e)}")
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
        """
        Ajoute ou retire un like sur un élément de la galerie

        Args:
            gallery_item_id (str): ID de l'élément de galerie
            user_id (str): ID de l'utilisateur qui like

        Returns:
            dict: Résultat de l'opération
        """
        try:
            # Vérifier si l'élément existe
            if not self.redis.exists(gallery_item_id):
                return {'success': False, 'error': 'Élément non trouvé'}

            # Clé pour suivre les likes des utilisateurs
            user_like_key = f"{gallery_item_id}:likes:{user_id}"

            pipe = self.redis.pipeline()

            # Vérifier si l'utilisateur a déjà liké cet élément
            already_liked = self.redis.exists(user_like_key)

            if already_liked:
                # Retirer le like
                pipe.delete(user_like_key)
                pipe.hincrby(gallery_item_id, 'likes', -1)
                pipe.execute()
                return {'success': True, 'action': 'unliked'}
            else:
                # Ajouter le like
                pipe.setex(user_like_key, 60 * 60 * 24 * 365, 1)  # TTL d'un an
                pipe.hincrby(gallery_item_id, 'likes', 1)
                pipe.execute()
                return {'success': True, 'action': 'liked'}

        except Exception as e:
            app.logger.error(f"Erreur lors du like de l'élément: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_gallery_item(self, gallery_item_id):
        """
        Récupère les détails d'un élément spécifique de la galerie

        Args:
            gallery_item_id (str): ID de l'élément de galerie

        Returns:
            dict: Données de l'élément de galerie
        """
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

            # Charger les métadonnées originales
            if 'original_metadata' in metadata_dict:
                try:
                    original_metadata = json.loads(metadata_dict['original_metadata'])

                    # Fusionner avec les métadonnées de galerie
                    item_data = {**metadata_dict, **original_metadata}

                    # Transformer la clé en URL
                    original_key = metadata_dict.get('original_key')
                    if original_key:
                        item_data['image_url'] = f"/image/{original_key}"

                    # Incrémenter le compteur de vues
                    self.redis.hincrby(gallery_item_id, 'views', 1)

                    return item_data

                except json.JSONDecodeError:
                    app.logger.error(
                        f"Erreur lors du décodage des métadonnées: {metadata_dict.get('original_metadata')}")
                    return None

            return metadata_dict

        except Exception as e:
            app.logger.error(f"Erreur lors de la récupération de l'élément: {str(e)}")
            return None

    def delete_from_gallery(self, gallery_item_id, user_id=None, is_admin=False):
        """
        Supprime un élément de la galerie (limité à l'utilisateur qui l'a partagé ou aux admins)

        Args:
            gallery_item_id (str): ID de l'élément de galerie
            user_id (str): ID de l'utilisateur qui demande la suppression
            is_admin (bool): Si l'utilisateur est un administrateur

        Returns:
            bool: Succès de l'opération
        """
        try:
            # Vérifier si l'élément existe
            if not self.redis.exists(gallery_item_id):
                return False

            # Vérifier les permissions
            if not is_admin:
                shared_by = self.redis.hget(gallery_item_id, 'shared_by')
                if shared_by:
                    shared_by = shared_by.decode('utf-8') if isinstance(shared_by, bytes) else shared_by
                    if shared_by != str(user_id):
                        return False

            # Récupérer les métadonnées pour les utiliser dans le nettoyage des indices
            metadata = self.redis.hgetall(gallery_item_id)
            metadata_dict = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                    v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in metadata.items()
            }

            # Utiliser un pipeline pour les opérations atomiques
            pipe = self.redis.pipeline()

            # Supprimer de l'index principal
            pipe.zrem(f"{self.prefix}:index", gallery_item_id)

            # Supprimer des indices de filtrage si nécessaire
            if 'original_metadata' in metadata_dict:
                try:
                    original_metadata = json.loads(metadata_dict['original_metadata'])

                    # Supprimer des indices par modèle
                    if 'model' in original_metadata:
                        model = original_metadata.get('model')
                        pipe.srem(f"{self.prefix}:filter:model:{model}", gallery_item_id)

                    # Supprimer d'autres indices selon les besoins
                    # Cette logique doit correspondre à celle utilisée dans add_to_gallery

                except json.JSONDecodeError:
                    app.logger.error(
                        f"Erreur lors du décodage des métadonnées pour suppression: {metadata_dict.get('original_metadata')}")

            # Supprimer l'élément lui-même
            pipe.delete(gallery_item_id)

            # Exécuter toutes les opérations
            pipe.execute()

            return True

        except Exception as e:
            app.logger.error(f"Erreur lors de la suppression de l'élément de la galerie: {str(e)}")
            return False

    def feature_gallery_item(self, gallery_item_id, featured=True):
        """
        Marque un élément comme mis en avant (à utiliser par les admins)

        Args:
            gallery_item_id (str): ID de l'élément de galerie
            featured (bool): État de mise en avant

        Returns:
            bool: Succès de l'opération
        """
        try:
            # Vérifier si l'élément existe
            if not self.redis.exists(gallery_item_id):
                return False

            # Mettre à jour l'attribut featured
            self.redis.hset(gallery_item_id, 'featured', str(featured))

            # Mettre à jour l'index de filtrage des éléments mis en avant si nécessaire
            if featured:
                self.redis.sadd(f"{self.prefix}:filter:featured", gallery_item_id)
            else:
                self.redis.srem(f"{self.prefix}:filter:featured", gallery_item_id)

            return True

        except Exception as e:
            app.logger.error(f"Erreur lors de la mise en avant de l'élément: {str(e)}")
            return False

    def get_featured_items(self, limit=10):
        """
        Récupère les éléments mis en avant pour la page d'accueil

        Args:
            limit (int): Nombre maximum d'éléments à récupérer

        Returns:
            list: Éléments mis en avant
        """
        try:
            # Récupérer les clés des éléments mis en avant
            featured_keys = self.redis.smembers(f"{self.prefix}:filter:featured")

            if not featured_keys:
                return []

            # Limiter le nombre d'éléments
            featured_keys = list(featured_keys)[:limit]

            # Utiliser un pipeline pour récupérer efficacement toutes les métadonnées
            pipe = self.redis.pipeline()

            for key in featured_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                pipe.hgetall(key)

            results = pipe.execute()

            featured_items = []

            for i, metadata in enumerate(results):
                if metadata:
                    key = featured_keys[i].decode('utf-8') if isinstance(featured_keys[i], bytes) else featured_keys[i]

                    # Décoder les valeurs bytes en strings
                    metadata_dict = {
                        k.decode('utf-8') if isinstance(k, bytes) else k:
                            v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in metadata.items()
                    }

                    # Charger les métadonnées originales
                    if 'original_metadata' in metadata_dict:
                        try:
                            original_metadata = json.loads(metadata_dict['original_metadata'])

                            # Fusionner avec les métadonnées de galerie
                            item_data = {**metadata_dict, **original_metadata}

                            # Transformer la clé en URL
                            original_key = metadata_dict.get('original_key')
                            if original_key:
                                item_data['image_url'] = f"/image/{original_key}"

                            # Ajouter un identifiant unique pour la galerie
                            item_data['gallery_id'] = key
                            item_data['size'] = 'large'  # Les éléments en vedette sont généralement plus grands

                            featured_items.append(item_data)

                        except json.JSONDecodeError:
                            app.logger.error(
                                f"Erreur lors du décodage des métadonnées: {metadata_dict.get('original_metadata')}")
                            continue

            return featured_items

        except Exception as e:
            app.logger.error(f"Erreur lors de la récupération des éléments mis en avant: {str(e)}")
            return []