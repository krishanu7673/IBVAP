import time

import numpy as np


class EntityRegistry:

    def __init__(
        self,
        similarity_threshold=0.80,
        max_entity_age=60
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entity_age = max_entity_age

        # entity_id -> entity information
        self.entities = {}

        # (camera_id, track_id) -> entity_id
        self.track_to_entity = {}

        self.next_entity_number = 1

    # --------------------------------------------------
    # CREATE ENTITY ID
    # --------------------------------------------------

    def _create_entity_id(self):
        entity_id = (
            f"E-{self.next_entity_number:04d}"
        )

        self.next_entity_number += 1

        return entity_id

    # --------------------------------------------------
    # FEATURE SIMILARITY
    # --------------------------------------------------

    def _cosine_similarity(
        self,
        feature_a,
        feature_b
    ):
        if feature_a is None or feature_b is None:
            return 0.0

        try:
            feature_a = np.asarray(
                feature_a,
                dtype=np.float32
            )

            feature_b = np.asarray(
                feature_b,
                dtype=np.float32
            )

            norm_a = np.linalg.norm(feature_a)
            norm_b = np.linalg.norm(feature_b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            similarity = np.dot(
                feature_a,
                feature_b
            ) / (
                norm_a * norm_b
            )

            return float(
                max(
                    0.0,
                    min(1.0, similarity)
                )
            )

        except Exception:
            return 0.0

    # --------------------------------------------------
    # UPDATE APPEARANCE FEATURE
    # --------------------------------------------------

    def _update_feature(
        self,
        entity,
        feature,
        alpha=0.20
    ):
        """
        Slowly update the entity appearance representation.

        This prevents a single noisy frame from completely
        replacing the established appearance feature.
        """

        if feature is None:
            return

        try:
            feature = np.asarray(
                feature,
                dtype=np.float32
            )

            if feature.size == 0:
                return

            existing_feature = entity.get("feature")

            if existing_feature is None:
                entity["feature"] = feature.copy()
                return

            existing_feature = np.asarray(
                existing_feature,
                dtype=np.float32
            )

            if existing_feature.shape != feature.shape:
                entity["feature"] = feature.copy()
                return

            updated_feature = (
                (1.0 - alpha) * existing_feature
                + alpha * feature
            )

            norm = np.linalg.norm(updated_feature)

            if norm > 0:
                updated_feature = (
                    updated_feature / norm
                )

            entity["feature"] = (
                updated_feature.astype(np.float32)
            )

        except Exception:
            pass

    # --------------------------------------------------
    # FIND EXISTING ENTITY
    # --------------------------------------------------

    def find_matching_entity(
        self,
        feature,
        camera_id,
        track_id,
        current_time=None
    ):
        if current_time is None:
            current_time = time.time()

        best_entity_id = None
        best_similarity = 0.0

        for entity_id, entity in self.entities.items():

            # --------------------------------------------------
            # IGNORE EXPIRED ENTITIES
            # --------------------------------------------------

            last_seen = entity.get(
                "last_seen",
                0
            )

            if (
                current_time - last_seen
                > self.max_entity_age
            ):
                continue

            # --------------------------------------------------
            # NEVER MATCH AN ENTITY TO ANOTHER TRACK
            # ON THE SAME CAMERA
            #
            # This is important because two people on the
            # same camera must not be merged solely from
            # appearance similarity.
            # --------------------------------------------------

            entity_current_camera = entity.get(
                "current_camera_id"
            )

            if entity_current_camera == camera_id:
                continue

            # --------------------------------------------------
            # FEATURE CHECK
            # --------------------------------------------------

            stored_feature = entity.get(
                "feature"
            )

            similarity = self._cosine_similarity(
                feature,
                stored_feature
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entity_id = entity_id

        # --------------------------------------------------
        # ACCEPT MATCH
        # --------------------------------------------------

        if (
            best_entity_id is not None
            and best_similarity
            >= self.similarity_threshold
        ):
            return (
                best_entity_id,
                best_similarity
            )

        return None, best_similarity

    # --------------------------------------------------
    # REGISTER OBSERVATION
    # --------------------------------------------------

    def register_observation(
        self,
        camera_id,
        track_id,
        feature,
        current_time=None
    ):
        if current_time is None:
            current_time = time.time()

        track_key = (
            camera_id,
            track_id
        )

        # --------------------------------------------------
        # EXISTING LOCAL TRACK
        # --------------------------------------------------

        existing_entity = (
            self.track_to_entity.get(
                track_key
            )
        )

        if existing_entity is not None:

            entity = self.entities.get(
                existing_entity
            )

            if entity is not None:

                entity["last_seen"] = current_time

                entity["current_camera_id"] = (
                    camera_id
                )

                entity["current_track_id"] = (
                    track_id
                )

                # Keep original camera/track information
                # available for historical reference.
                if camera_id not in entity["cameras_seen"]:
                    entity["cameras_seen"].append(
                        camera_id
                    )

                self._update_feature(
                    entity,
                    feature
                )

                entity["observations"] = (
                    entity.get(
                        "observations",
                        0
                    ) + 1
                )

                entity["last_similarity"] = 1.0

                return {
                    "entity_id": existing_entity,
                    "similarity": 1.0,
                    "is_new": False
                }

            # If the entity disappeared unexpectedly,
            # remove the stale track mapping.
            self.track_to_entity.pop(
                track_key,
                None
            )

        # --------------------------------------------------
        # SEARCH FOR CROSS-CAMERA MATCH
        # --------------------------------------------------

        entity_id, similarity = (
            self.find_matching_entity(
                feature,
                camera_id,
                track_id,
                current_time
            )
        )

        # --------------------------------------------------
        # EXISTING GLOBAL ENTITY FOUND
        # --------------------------------------------------

        if entity_id is not None:

            self.track_to_entity[
                track_key
            ] = entity_id

            entity = self.entities[
                entity_id
            ]

            entity["last_seen"] = current_time

            entity["current_camera_id"] = (
                camera_id
            )

            entity["current_track_id"] = (
                track_id
            )

            # --------------------------------------------------
            # CAMERA HISTORY
            # --------------------------------------------------

            if camera_id not in entity["cameras_seen"]:
                entity["cameras_seen"].append(
                    camera_id
                )

            # --------------------------------------------------
            # TRACK HISTORY
            # --------------------------------------------------

            entity["track_history"].append(
                {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "timestamp": current_time
                }
            )

            # --------------------------------------------------
            # APPEARANCE UPDATE
            # --------------------------------------------------

            self._update_feature(
                entity,
                feature
            )

            entity["observations"] = (
                entity.get(
                    "observations",
                    0
                ) + 1
            )

            entity["last_similarity"] = (
                similarity
            )

            entity["cross_camera_matches"] = (
                entity.get(
                    "cross_camera_matches",
                    0
                ) + 1
            )

            return {
                "entity_id": entity_id,
                "similarity": similarity,
                "is_new": False
            }

        # --------------------------------------------------
        # CREATE NEW GLOBAL ENTITY
        # --------------------------------------------------

        entity_id = self._create_entity_id()

        self.entities[entity_id] = {

            "entity_id": entity_id,

            # Current location
            "current_camera_id": camera_id,
            "current_track_id": track_id,

            # Historical compatibility fields
            "camera_id": camera_id,
            "track_id": track_id,

            # Appearance
            "feature": (
                np.asarray(
                    feature,
                    dtype=np.float32
                ).copy()
                if feature is not None
                else None
            ),

            # Lifecycle
            "first_seen": current_time,
            "last_seen": current_time,

            # Statistics
            "observations": 1,
            "last_similarity": 0.0,
            "cross_camera_matches": 0,

            # Camera history
            "cameras_seen": [
                camera_id
            ],

            # Track history
            "track_history": [
                {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "timestamp": current_time
                }
            ]
        }

        self.track_to_entity[
            track_key
        ] = entity_id

        return {
            "entity_id": entity_id,
            "similarity": 0.0,
            "is_new": True
        }

    # --------------------------------------------------
    # GET ENTITY FOR TRACK
    # --------------------------------------------------

    def get_entity_for_track(
        self,
        camera_id,
        track_id
    ):
        return self.track_to_entity.get(
            (
                camera_id,
                track_id
            )
        )

    # --------------------------------------------------
    # GET ENTITY
    # --------------------------------------------------

    def get_entity(
        self,
        entity_id
    ):
        return self.entities.get(
            entity_id
        )

    # --------------------------------------------------
    # GET ALL ENTITIES
    # --------------------------------------------------

    def get_all_entities(self):
        return self.entities.copy()

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def cleanup(self):

        current_time = time.time()

        expired_entities = []

        # --------------------------------------------------
        # FIND EXPIRED ENTITIES
        # --------------------------------------------------

        for entity_id, entity in self.entities.items():

            last_seen = entity.get(
                "last_seen",
                0
            )

            if (
                current_time - last_seen
                > self.max_entity_age
            ):
                expired_entities.append(
                    entity_id
                )

        # --------------------------------------------------
        # REMOVE EXPIRED ENTITIES
        # --------------------------------------------------

        for entity_id in expired_entities:

            self.entities.pop(
                entity_id,
                None
            )

            # --------------------------------------------------
            # IMPORTANT:
            # Remove EVERY track mapping belonging to
            # this global entity, not just the latest one.
            # --------------------------------------------------

            stale_tracks = [
                track_key
                for track_key, mapped_entity_id
                in self.track_to_entity.items()
                if mapped_entity_id == entity_id
            ]

            for track_key in stale_tracks:
                self.track_to_entity.pop(
                    track_key,
                    None
                )