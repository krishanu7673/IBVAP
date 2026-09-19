import os
import time
import cv2


class EvidenceManager:

    def __init__(
        self,
        base_directory="evidence"
    ):

        self.base_directory = (
            base_directory
        )

        self.snapshot_directory = os.path.join(
            self.base_directory,
            "snapshots"
        )

        self.clip_directory = os.path.join(
            self.base_directory,
            "clips"
        )

        os.makedirs(
            self.snapshot_directory,
            exist_ok=True
        )

        os.makedirs(
            self.clip_directory,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # SAVE SNAPSHOT
    # ---------------------------------------------------------

    def save_snapshot(
        self,
        frame,
        camera_id,
        event_type,
        track_id=None
    ):

        if frame is None:
            return None

        timestamp = int(
            time.time() * 1000
        )

        safe_camera = str(
            camera_id
        ).replace(
            " ",
            "_"
        )

        safe_event = str(
            event_type
        ).replace(
            " ",
            "_"
        )

        filename = (
            f"{safe_camera}_"
            f"{safe_event}_"
            f"{track_id}_"
            f"{timestamp}.jpg"
        )

        filepath = os.path.join(
            self.snapshot_directory,
            filename
        )

        success = cv2.imwrite(
            filepath,
            frame
        )

        if not success:
            return None

        return filepath

    # ---------------------------------------------------------
    # SAVE MANUAL EVIDENCE
    # ---------------------------------------------------------

    def save_manual_snapshot(
        self,
        frame,
        camera_id
    ):

        return self.save_snapshot(
            frame,
            camera_id,
            "MANUAL_CAPTURE"
        )