import cv2
import numpy as np

from src.entity_registry import (
    EntityRegistry
)


class CrossCameraCorrelator:

    def __init__(
        self,
        similarity_threshold=0.80,
        max_entity_age=60
    ):

        self.registry = EntityRegistry(
            similarity_threshold=similarity_threshold,
            max_entity_age=max_entity_age
        )

    # --------------------------------------------------
    # EXTRACT APPEARANCE FEATURE
    # --------------------------------------------------

    def extract_feature(
        self,
        frame,
        bbox
    ):

        if frame is None:
            return None

        if bbox is None:
            return None

        try:

            x1, y1, x2, y2 = bbox

            height, width = frame.shape[:2]

            x1 = max(
                0,
                min(int(x1), width - 1)
            )

            y1 = max(
                0,
                min(int(y1), height - 1)
            )

            x2 = max(
                0,
                min(int(x2), width)
            )

            y2 = max(
                0,
                min(int(y2), height)
            )

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                return None

            # --------------------------------------------------
            # RESIZE
            # --------------------------------------------------

            crop = cv2.resize(
                crop,
                (64, 128),
                interpolation=cv2.INTER_AREA
            )

            # --------------------------------------------------
            # HSV
            # --------------------------------------------------

            hsv = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2HSV
            )

            h_hist = cv2.calcHist(
                [hsv],
                [0],
                None,
                [32],
                [0, 180]
            )

            s_hist = cv2.calcHist(
                [hsv],
                [1],
                None,
                [32],
                [0, 256]
            )

            v_hist = cv2.calcHist(
                [hsv],
                [2],
                None,
                [32],
                [0, 256]
            )

            # --------------------------------------------------
            # GRAYSCALE
            # --------------------------------------------------

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            gray_hist = cv2.calcHist(
                [gray],
                [0],
                None,
                [32],
                [0, 256]
            )

            # --------------------------------------------------
            # NORMALIZATION
            # --------------------------------------------------

            cv2.normalize(
                h_hist,
                h_hist
            )

            cv2.normalize(
                s_hist,
                s_hist
            )

            cv2.normalize(
                v_hist,
                v_hist
            )

            cv2.normalize(
                gray_hist,
                gray_hist
            )

            # --------------------------------------------------
            # COMBINE
            # --------------------------------------------------

            feature = np.concatenate(
                [
                    h_hist.flatten(),
                    s_hist.flatten(),
                    v_hist.flatten(),
                    gray_hist.flatten()
                ]
            )

            norm = np.linalg.norm(
                feature
            )

            if norm > 0:
                feature = (
                    feature / norm
                )

            return feature.astype(
                np.float32
            )

        except Exception:
            return None

    # --------------------------------------------------
    # PROCESS DETECTION
    # --------------------------------------------------

    def process_detection(
        self,
        frame,
        camera_id,
        track_id,
        bbox
    ):

        if track_id is None:
            return None

        if track_id < 0:
            return None

        feature = self.extract_feature(
            frame,
            bbox
        )

        if feature is None:
            return None

        result = (
            self.registry.register_observation(
                camera_id=camera_id,
                track_id=track_id,
                feature=feature
            )
        )

        similarity = result.get(
            "similarity",
            0.0
        )

        return {
            "entity_id": result.get(
                "entity_id"
            ),

            "similarity": round(
                similarity * 100,
                2
            ),

            "is_new": result.get(
                "is_new",
                False
            ),

            "possible_same_entity": (
                similarity
                >= self.registry.similarity_threshold
            )
        }
        # --------------------------------------------------
    # PROCESS MULTIPLE DETECTIONS
    # --------------------------------------------------
    def process_detections(
        self,
        frame,
        camera_id,
        detections
    ):
        """
        Process all person detections in the current frame.

        Uses the existing process_detection() method for
        each valid tracked person.

        Returns the original detection list with
        anonymous entity information attached.
        """

        if frame is None:
            return detections

        if detections is None:
            return []

        for detection in detections:

            if not isinstance(detection, dict):
                continue

            # --------------------------------------------------
            # Only correlate persons
            # --------------------------------------------------
            if detection.get("label") != "person":
                continue

            track_id = detection.get("track_id")

            if track_id is None or track_id < 0:
                continue

            bbox = detection.get("bbox")

            if bbox is None:
                continue

            try:
                result = self.process_detection(
                    frame=frame,
                    camera_id=camera_id,
                    track_id=track_id,
                    bbox=bbox
                )

                if result is None:
                    continue

                # --------------------------------------------------
                # Attach anonymous entity information
                # --------------------------------------------------
                detection["entity_id"] = result.get(
                    "entity_id"
                )

                detection["entity_similarity"] = result.get(
                    "similarity",
                    0.0
                )

                detection["entity_is_new"] = result.get(
                    "is_new",
                    False
                )

                detection["possible_same_entity"] = result.get(
                    "possible_same_entity",
                    False
                )

            except Exception as error:
                print(
                    f"[ENTITY] Processing error for "
                    f"Track {track_id}: {error}"
                )

        return detections
    # --------------------------------------------------
    # GET ENTITY
    # --------------------------------------------------

    def get_entity(
        self,
        camera_id,
        track_id
    ):

        return (
            self.registry.get_entity_for_track(
                camera_id,
                track_id
            )
        )

    # --------------------------------------------------
    # GET ENTITY DETAILS
    # --------------------------------------------------

    def get_entity_details(
        self,
        entity_id
    ):

        return (
            self.registry.get_entity(
                entity_id
            )
        )

    # --------------------------------------------------
    # GET ALL ENTITIES
    # --------------------------------------------------

    def get_all_entities(self):

        return (
            self.registry.get_all_entities()
        )

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def cleanup(self):

        self.registry.cleanup()