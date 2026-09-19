import time

import cv2
import numpy as np

from ultralytics import YOLO

from src.intrusion import (
    PerimeterIntrusionDetector
)

from src.anpr import ANPREngine

from src.behavior import (
    BehaviorAnalyzer
)

from src.spatial_intelligence import (
    SpatialIntelligence
)


class BorderAnalyticsEngine:

    VEHICLE_CLASSES = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike"
    }

    PERSON_CLASS = "person"

    def __init__(
        self,
        model_path="yolov8n.onnx",
        use_gpu=True,
        perimeter_line=((50, 400), (600, 400)),
        loitering_seconds=20.0,
        confidence_threshold=0.35,
        iou_threshold=0.50,
        zones=None
    ):

        self.model_path = model_path
        self.use_gpu = use_gpu

        # ==========================================================
        # SPATIAL INTELLIGENCE
        # ==========================================================

        self.spatial_intelligence = SpatialIntelligence(
            zones=zones or []
        )

        print(
            f"[*] Spatial intelligence initialized with "
            f"{len(zones or [])} zone(s)."
        )

        # ==========================================================
        # DETECTION CONFIGURATION
        # ==========================================================

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.iou_threshold = float(
            iou_threshold
        )

        self.confidence_threshold = max(
            0.0,
            min(
                self.confidence_threshold,
                1.0
            )
        )

        self.iou_threshold = max(
            0.0,
            min(
                self.iou_threshold,
                1.0
            )
        )

        # ==========================================================
        # PERFORMANCE MONITORING
        # ==========================================================

        self.performance_frame_count = 0
        self.performance_start_time = time.time()
        self.last_processing_time = 0.0
        self.average_processing_time = 0.0
        self.processing_fps = 0.0

        # ==========================================================
        # DEVICE
        # ==========================================================

        if use_gpu:
            self.device = 0

            # OCR remains on CPU because of the current
            # PaddleOCR compatibility configuration.
            ocr_device = "cpu"

        else:
            self.device = "cpu"
            ocr_device = "cpu"

        # ==========================================================
        # YOLO
        # ==========================================================

        print(
            f"[*] Loading YOLO model: {model_path}"
        )

        self.detector = YOLO(
            model_path,
            task="detect"
        )

        print(
            f"[*] YOLO device: {self.device}"
        )

        print(
            f"[*] Detection confidence: "
            f"{self.confidence_threshold:.2f}"
        )

        print(
            f"[*] Detection IoU: "
            f"{self.iou_threshold:.2f}"
        )

        # ==========================================================
        # PERIMETER INTRUSION
        # ==========================================================

        self.intrusion_detector = (
            PerimeterIntrusionDetector(
                line_start=perimeter_line[0],
                line_end=perimeter_line[1],
                cooldown_seconds=5.0
            )
        )

        # ==========================================================
        # BEHAVIOR ANALYZER
        # ==========================================================

        self.behavior_analyzer = BehaviorAnalyzer(
            perimeter_line=perimeter_line,
            restricted_distance=150,
            high_security_distance=75,
            loitering_seconds=loitering_seconds,
            early_warning_score=60,
            repeated_approach_threshold=2
        )

        print(
            "[*] Behavior analysis engine initialized."
        )

        print(
            "[*] Early-warning intelligence enabled."
        )

        # ==========================================================
        # ANPR
        # ==========================================================

        print(
            "[*] Loading ANPR engine..."
        )

        self.anpr = ANPREngine(
            confidence_threshold=0.60,
            device=ocr_device
        )

        self.loitering_seconds = (
            float(loitering_seconds)
        )

        # ==========================================================
        # CAMERA-AWARE TRACK STATE
        # ==========================================================

        # Keys use:
        # (camera_id, track_id)
        #
        # This prevents Track ID collisions between cameras.

        self.track_start_time = {}
        self.last_plate_check = {}
        self.plate_cache = {}

        # ==========================================================
        # TRACK LIFECYCLE INTELLIGENCE
        # ==========================================================

        self.track_lifecycle = {}

        # A temporary tracking dropout should not immediately
        # destroy the historical state of the track.

        self.track_timeout_seconds = 3.0

    # ==============================================================
    # APPEARANCE FEATURE EXTRACTION
    # ==============================================================

    def _extract_appearance_feature(
        self,
        frame,
        bbox
    ):
        """
        Generate a lightweight anonymous appearance descriptor
        for future cross-camera entity correlation.

        This does NOT identify a person.

        It describes visual appearance using color and
        grayscale distributions.
        """

        if frame is None or bbox is None:
            return None

        try:

            x1, y1, x2, y2 = bbox

            h, w = frame.shape[:2]

            x1 = max(
                0,
                min(int(x1), w - 1)
            )

            y1 = max(
                0,
                min(int(y1), h - 1)
            )

            x2 = max(
                0,
                min(int(x2), w)
            )

            y2 = max(
                0,
                min(int(y2), h)
            )

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                return None

            crop = cv2.resize(
                crop,
                (64, 128),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------------
            # Convert to HSV
            # ------------------------------------------------------

            hsv = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2HSV
            )

            # ------------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------------

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------------
            # Hue histogram
            # ------------------------------------------------------

            hist_h = cv2.calcHist(
                [hsv],
                [0],
                None,
                [16],
                [0, 180]
            )

            # ------------------------------------------------------
            # Saturation histogram
            # ------------------------------------------------------

            hist_s = cv2.calcHist(
                [hsv],
                [1],
                None,
                [16],
                [0, 256]
            )

            # ------------------------------------------------------
            # Value histogram
            # ------------------------------------------------------

            hist_v = cv2.calcHist(
                [hsv],
                [2],
                None,
                [16],
                [0, 256]
            )

            # ------------------------------------------------------
            # Grayscale histogram
            # ------------------------------------------------------

            hist_gray = cv2.calcHist(
                [gray],
                [0],
                None,
                [16],
                [0, 256]
            )

            # ------------------------------------------------------
            # Combine features
            # ------------------------------------------------------

            feature = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten(),
                hist_gray.flatten()
            ]).astype(
                np.float32
            )

            # ------------------------------------------------------
            # L2 normalization
            # ------------------------------------------------------

            norm = np.linalg.norm(
                feature
            )

            if norm > 0:
                feature /= norm

            return feature

        except Exception as error:

            print(
                f"[ENTITY] Appearance feature error: "
                f"{error}"
            )

            return None

    # ==============================================================
    # PERFORMANCE MONITORING
    # ==============================================================

    def _update_performance(
        self,
        processing_time
    ):

        self.performance_frame_count += 1

        self.last_processing_time = (
            processing_time
        )

        # Running average processing time.

        if self.performance_frame_count == 1:

            self.average_processing_time = (
                processing_time
            )

        else:

            self.average_processing_time = (
                (
                    self.average_processing_time
                    * (
                        self.performance_frame_count - 1
                    )
                )
                + processing_time
            ) / self.performance_frame_count

        elapsed = (
            time.time()
            - self.performance_start_time
        )

        if elapsed > 0:

            self.processing_fps = (
                self.performance_frame_count
                / elapsed
            )

    def get_performance(self):

        return {
            "frames_processed":
                self.performance_frame_count,

            "fps":
                round(
                    self.processing_fps,
                    2
                ),

            "last_processing_time_ms":
                round(
                    self.last_processing_time * 1000,
                    2
                ),

            "average_processing_time_ms":
                round(
                    self.average_processing_time * 1000,
                    2
                )
        }

    # ==============================================================
    # LOITERING LOGIC
    # ==============================================================

    def _check_loitering(
        self,
        camera_id,
        track_id
    ):

        if track_id is None or track_id < 0:
            return False, 0.0

        key = (
            str(camera_id),
            int(track_id)
        )

        now = time.time()

        if key not in self.track_start_time:

            self.track_start_time[key] = now

            return False, 0.0

        duration = (
            now
            - self.track_start_time[key]
        )

        return (
            duration >= self.loitering_seconds,
            duration
        )

    # ==============================================================
    # TRACK LIFECYCLE INTELLIGENCE
    # ==============================================================

    def _update_track_lifecycle(
        self,
        camera_id,
        track_id,
        confidence
    ):

        if track_id is None or track_id < 0:
            return None

        key = (
            str(camera_id),
            int(track_id)
        )

        now = time.time()

        # ----------------------------------------------------------
        # First observation
        # ----------------------------------------------------------

        if key not in self.track_lifecycle:

            self.track_lifecycle[key] = {

                "camera_id":
                    camera_id,

                "track_id":
                    track_id,

                "first_seen":
                    now,

                "last_seen":
                    now,

                "detection_count":
                    1,

                "confidence_sum":
                    float(confidence),

                "average_confidence":
                    float(confidence)
            }

        # ----------------------------------------------------------
        # Existing track
        # ----------------------------------------------------------

        else:

            state = self.track_lifecycle[key]

            state["last_seen"] = now

            state["detection_count"] += 1

            state["confidence_sum"] += (
                float(confidence)
            )

            state["average_confidence"] = (
                state["confidence_sum"]
                / state["detection_count"]
            )

        # ----------------------------------------------------------
        # Current duration
        # ----------------------------------------------------------

        state = self.track_lifecycle[key]

        duration = (
            state["last_seen"]
            - state["first_seen"]
        )

        state["duration"] = duration

        return {

            "camera_id":
                camera_id,

            "track_id":
                track_id,

            "first_seen":
                state["first_seen"],

            "last_seen":
                state["last_seen"],

            "duration":
                duration,

            "detection_count":
                state["detection_count"],

            "average_confidence":
                state["average_confidence"]
        }

    def get_track_lifecycle(
        self,
        camera_id=None,
        track_id=None
    ):

        # ----------------------------------------------------------
        # Return one specific camera + track
        # ----------------------------------------------------------

        if (
            camera_id is not None
            and track_id is not None
        ):

            key = (
                str(camera_id),
                int(track_id)
            )

            state = self.track_lifecycle.get(
                key
            )

            if state is None:
                return None

            return state.copy()

        # ----------------------------------------------------------
        # Return all lifecycle states
        # ----------------------------------------------------------

        result = {}

        for key, state in (
            self.track_lifecycle.items()
        ):

            result[key] = state.copy()

        return result

    # ==============================================================
    # TRACK CLEANUP
    # ==============================================================

    def _cleanup_tracks(
        self,
        camera_id,
        active_track_ids
    ):

        active_track_ids = {
            int(track_id)
            for track_id in active_track_ids
            if track_id is not None
        }

        camera_id = str(camera_id)

        # ----------------------------------------------------------
        # Existing track state cleanup
        # ----------------------------------------------------------

        for key in list(
            self.track_start_time.keys()
        ):

            key_camera, key_track = key

            if (
                key_camera == camera_id
                and key_track not in active_track_ids
            ):

                self.track_start_time.pop(
                    key,
                    None
                )

                self.last_plate_check.pop(
                    key,
                    None
                )

                self.plate_cache.pop(
                    key,
                    None
                )

        # ----------------------------------------------------------
        # Track lifecycle cleanup
        # ----------------------------------------------------------

        current_time = time.time()

        for key in list(
            self.track_lifecycle.keys()
        ):

            key_camera, key_track = key

            if key_camera != camera_id:
                continue

            state = self.track_lifecycle[key]

            last_seen = state[
                "last_seen"
            ]

            # Preserve the track for a short period after
            # it disappears from the current detection frame.

            if (
                current_time - last_seen
                > self.track_timeout_seconds
            ):

                self.track_lifecycle.pop(
                    key,
                    None
                )

        # ----------------------------------------------------------
        # Intrusion detector cleanup
        # ----------------------------------------------------------

        self.intrusion_detector.cleanup(
            active_track_ids
        )

        # ----------------------------------------------------------
        # Spatial intelligence cleanup
        # ----------------------------------------------------------

        self.spatial_intelligence.cleanup(
            active_track_ids
        )

        # ----------------------------------------------------------
        # Behavior analyzer cleanup
        # ----------------------------------------------------------

        self.behavior_analyzer.cleanup(
            max_age=30
        )

    # ==============================================================
    # EVENT BUILDER
    # ==============================================================

    def _build_event(
        self,
        camera_id,
        event_type,
        details,
        track_id=None,
        confidence=None,
        behavior=None,
        extra=None
    ):

        if behavior is None:
            behavior = {}

        event = {

            "type":
                event_type,

            "event_type":
                event_type,

            "camera_id":
                camera_id,

            "track_id":
                track_id,

            "confidence":
                confidence,

            "details":
                details,

            "timestamp":
                time.time(),

            "risk_score":
                behavior.get(
                    "risk_score",
                    0
                ),

            "risk_level":
                behavior.get(
                    "risk_level",
                    "LOW"
                ),

            "zone":
                behavior.get(
                    "zone",
                    "UNKNOWN"
                ),

            "direction":
                behavior.get(
                    "direction",
                    "UNKNOWN"
                ),

            "distance_to_perimeter":
                behavior.get(
                    "distance_to_perimeter"
                ),

            "dwell_time":
                behavior.get(
                    "dwell_time",
                    0.0
                ),

            "approaching":
                behavior.get(
                    "approaching",
                    False
                ),

            "risk_trend":
                behavior.get(
                    "risk_trend",
                    "STABLE"
                ),

            "risk_escalating":
                behavior.get(
                    "risk_escalating",
                    False
                ),

            "risk_reasons":
                behavior.get(
                    "risk_reasons",
                    []
                ),

            # NEW EARLY-WARNING FIELDS

            "early_warning":
                behavior.get(
                    "early_warning",
                    False
                ),

            "early_warning_reasons":
                behavior.get(
                    "early_warning_reasons",
                    []
                ),

            "approach_count":
                behavior.get(
                    "approach_count",
                    0
                ),

            "approach_duration":
                behavior.get(
                    "approach_duration",
                    0.0
                ),

            "risk_increase_streak":
                behavior.get(
                    "risk_increase_streak",
                    0
                )
        }

        if extra:

            event.update(
                extra
            )

        return event

    # ==============================================================
    # PROCESS FRAME
    # ==============================================================

    def process_frame(
        self,
        frame,
        camera_id
    ):

        processing_start = time.time()

        # ==========================================================
        # INVALID FRAME PROTECTION
        # ==========================================================

        if frame is None:
            return [], []

        try:

            results = self.detector.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=self.device,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

        except Exception as error:

            print(
                f"[AI] Inference error on "
                f"{camera_id}: {error}"
            )

            return [], []

        # ==========================================================
        # UPDATE PERFORMANCE
        # ==========================================================

        processing_time = (
            time.time()
            - processing_start
        )

        self._update_performance(
            processing_time
        )

        if not results:
            return [], []

        result = results[0]

        detections = []
        events = []

        active_track_ids = []

        if result.boxes is None:
            return detections, events

        boxes = result.boxes

        for index in range(
            len(boxes)
        ):

            box = boxes[index]

            # ======================================================
            # READ DETECTION
            # ======================================================

            try:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().numpy()
                )

                cls_id = int(
                    box.cls[0].item()
                )

                conf = float(
                    box.conf[0].item()
                )

            except Exception as error:

                print(
                    f"[AI] Invalid detection "
                    f"on {camera_id}: {error}"
                )

                continue

            # ======================================================
            # CLASS LABEL
            # ======================================================

            try:

                label = self.detector.names[
                    cls_id
                ]

            except Exception:

                label = str(
                    cls_id
                )

            # ======================================================
            # TRACK ID
            # ======================================================

            track_id = -1

            if boxes.id is not None:

                try:

                    track_id = int(
                        boxes.id[index].item()
                    )

                except Exception:

                    track_id = -1

            # ======================================================
            # TRACK LIFECYCLE
            # ======================================================

            lifecycle = None

            if track_id >= 0:

                active_track_ids.append(
                    track_id
                )

                lifecycle = (
                    self._update_track_lifecycle(
                        camera_id,
                        track_id,
                        conf
                    )
                )

            # ======================================================
            # CLAMP COORDINATES
            # ======================================================

            h, w = frame.shape[:2]

            x1 = max(
                0,
                min(x1, w - 1)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            # Use w/h as the upper boundary for ROI slicing.

            x2 = max(
                0,
                min(x2, w)
            )

            y2 = max(
                0,
                min(y2, h)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            event = None

            plate_text = None

            plate_conf = 0.0

            appearance_feature = None
            spatial = {
                "zone_id": "OUTSIDE_DEFINED_ZONES",
                "zone_name": "Outside Defined Zones",
                "zone_type": "UNKNOWN",
                "risk_weight": 0,
                "previous_zone_id": None,
                "zone_changed": False,
                "entered_zone": False,
                "exited_zone": False,
                "dwell_time": 0.0
            }

            if track_id >= 0:
                try:
                    spatial_result = self.spatial_intelligence.analyze(
                        track_id=track_id,
                        center=center
                    )
                    if isinstance(spatial_result, dict):
                        spatial.update(spatial_result)
                except Exception as error:
                    print(
                        f"[SPATIAL] Error for Track {track_id}: {error}"
                    )

            # ======================================================
            # BEHAVIOR DATA DEFAULTS
            # ======================================================

            behavior = {

                "track_id":
                    track_id,

                "zone":
                    "UNKNOWN",

                "direction":
                    "UNKNOWN",

                "distance_to_perimeter":
                    None,

                "dwell_time":
                    0.0,

                "approaching":
                    False,

                "loitering":
                    False,

                "risk_score":
                    0,

                "risk_level":
                    "LOW",

                "risk_trend":
                    "STABLE",

                "risk_escalating":
                    False,

                "risk_reasons":
                    [],

                # NEW

                "early_warning":
                    False,

                "early_warning_reasons":
                    [],

                "approach_count":
                    0,

                "approach_duration":
                    0.0,

                "risk_increase_streak":
                    0
            }

            # ======================================================
            # SPATIAL ZONE INTELLIGENCE
            # ======================================================

            if track_id >= 0 and spatial.get("zone_changed", False):
                previous_zone = spatial.get(
                    "previous_zone_id",
                    "UNKNOWN"
                )
                current_zone = spatial.get(
                    "zone_id",
                    "UNKNOWN"
                )
                zone_name = spatial.get(
                    "zone_name",
                    current_zone
                )

                if spatial.get("entered_zone", False):
                    spatial_event_type = "ZONE_ENTRY"
                    spatial_details = (
                        f"Track ID {track_id} entered zone "
                        f"{zone_name} ({current_zone}) from "
                        f"{previous_zone}"
                    )
                elif spatial.get("exited_zone", False):
                    spatial_event_type = "ZONE_EXIT"
                    spatial_details = (
                        f"Track ID {track_id} exited defined zones "
                        f"from {previous_zone}"
                    )
                else:
                    spatial_event_type = "ZONE_TRANSITION"
                    spatial_details = (
                        f"Track ID {track_id} transitioned from "
                        f"{previous_zone} to {current_zone} "
                        f"({zone_name})"
                    )

                spatial_event = self._build_event(
                    camera_id=camera_id,
                    event_type=spatial_event_type,
                    details=spatial_details,
                    track_id=track_id,
                    confidence=conf,
                    behavior=behavior,
                    extra={
                        "zone_id": current_zone,
                        "zone_name": zone_name,
                        "zone_type": spatial.get(
                            "zone_type",
                            "UNKNOWN"
                        ),
                        "previous_zone_id": previous_zone,
                        "zone_dwell_time": spatial.get(
                            "dwell_time",
                            0.0
                        ),
                        "spatial_risk_weight": spatial.get(
                            "risk_weight",
                            0
                        )
                    }
                )

                events.append(spatial_event)

            # ======================================================
            # PERSON ANALYTICS
            # ======================================================

            if label == self.PERSON_CLASS:

                if track_id >= 0:

                    # ------------------------------------------------
                    # CROSS-CAMERA APPEARANCE FEATURE
                    # ------------------------------------------------

                    if lifecycle is not None:

                        detection_count = (
                            lifecycle.get(
                                "detection_count",
                                1
                            )
                        )

                        # Extract on first observation and
                        # periodically afterward.

                        if (
                            detection_count == 1
                            or
                            detection_count % 15 == 0
                        ):

                            appearance_feature = (
                                self._extract_appearance_feature(
                                    frame,
                                    (
                                        x1,
                                        y1,
                                        x2,
                                        y2
                                    )
                                )
                            )

                    # ------------------------------------------------
                    # BEHAVIOR ANALYSIS
                    # ------------------------------------------------

                    try:

                        behavior_result = (
                            self.behavior_analyzer.analyze(
                                track_id=track_id,
                                center=center
                            )
                        )

                        if isinstance(
                            behavior_result,
                            dict
                        ):

                            behavior.update(
                                behavior_result
                            )

                    except Exception as error:

                        print(
                            f"[BEHAVIOR] Error for "
                            f"Track {track_id}: "
                            f"{error}"
                        )

                    # ------------------------------------------------
                    # EARLY-WARNING EVENT
                    # ------------------------------------------------

                    if behavior.get(
                        "early_warning",
                        False
                    ):

                        warning_reasons = (
                            behavior.get(
                                "early_warning_reasons",
                                []
                            )
                        )

                        if not warning_reasons:

                            warning_reasons = [
                                "Elevated behavioral risk"
                            ]

                        reason_text = "; ".join(
                            str(reason)
                            for reason in warning_reasons
                        )

                        early_warning_event = (
                            self._build_event(
                                camera_id=camera_id,

                                event_type=
                                    "EARLY_WARNING",

                                details=(
                                    f"Person ID "
                                    f"{track_id} "
                                    f"early-warning state: "
                                    f"risk "
                                    f"{behavior.get('risk_score', 0)}/100, "
                                    f"zone: "
                                    f"{behavior.get('zone', 'UNKNOWN')}, "
                                    f"direction: "
                                    f"{behavior.get('direction', 'UNKNOWN')}, "
                                    f"reasons: "
                                    f"{reason_text}"
                                ),

                                track_id=track_id,

                                confidence=conf,

                                behavior=behavior,

                                extra={

                                    "early_warning":
                                        True,

                                    "early_warning_reasons":
                                        warning_reasons,

                                    "approach_count":
                                        behavior.get(
                                            "approach_count",
                                            0
                                        ),

                                    "approach_duration":
                                        behavior.get(
                                            "approach_duration",
                                            0.0
                                        ),

                                    "risk_increase_streak":
                                        behavior.get(
                                            "risk_increase_streak",
                                            0
                                        )
                                }
                            )
                        )

                        events.append(
                            early_warning_event
                        )

                    # ------------------------------------------------
                    # PERIMETER CROSSING
                    # ------------------------------------------------

                    try:

                        crossed = (
                            self.intrusion_detector
                            .check_crossing(
                                track_id,
                                center
                            )
                        )

                    except Exception as error:

                        print(
                            f"[INTRUSION] Error for "
                            f"Track {track_id}: "
                            f"{error}"
                        )

                        crossed = False

                    if crossed:

                        event = self._build_event(
                            camera_id=camera_id,

                            event_type=
                                "PERIMETER_BREACH",

                            details=(
                                f"Person ID "
                                f"{track_id} crossed "
                                f"the virtual perimeter"
                            ),

                            track_id=track_id,

                            confidence=conf,

                            behavior=behavior
                        )

                        events.append(
                            event
                        )

                    # ------------------------------------------------
                    # LOITERING
                    # ------------------------------------------------

                    loitering, duration = (
                        self._check_loitering(
                            camera_id,
                            track_id
                        )
                    )

                    if loitering:

                        loiter_event = (
                            self._build_event(

                                camera_id=camera_id,

                                event_type=
                                    "LOITERING_DETECTED",

                                details=(
                                    f"Person ID "
                                    f"{track_id} remained "
                                    f"in the monitored area "
                                    f"for "
                                    f"{duration:.1f} seconds"
                                ),

                                track_id=track_id,

                                confidence=conf,

                                behavior=behavior,

                                extra={

                                    "loitering_duration":
                                        duration
                                }
                            )
                        )

                        events.append(
                            loiter_event
                        )

                    # ------------------------------------------------
                    # BEHAVIOR-BASED RISK EVENT
                    # ------------------------------------------------

                    risk_level = behavior.get(
                        "risk_level",
                        "LOW"
                    )

                    if risk_level in (
                        "HIGH",
                        "CRITICAL"
                    ):

                        risk_reasons = (
                            behavior.get(
                                "risk_reasons",
                                []
                            )
                        )

                        reason_text = "; ".join(
                            str(reason)
                            for reason in risk_reasons
                        )

                        risk_event = (
                            self._build_event(

                                camera_id=camera_id,

                                event_type=
                                    "BEHAVIOR_RISK",

                                details=(
                                    f"Person ID "
                                    f"{track_id} "
                                    f"behavior risk "
                                    f"{risk_level} "
                                    f"({behavior.get('risk_score', 0)}/100), "
                                    f"zone: "
                                    f"{behavior.get('zone', 'UNKNOWN')}, "
                                    f"direction: "
                                    f"{behavior.get('direction', 'UNKNOWN')}"
                                ),

                                track_id=track_id,

                                confidence=conf,

                                behavior=behavior,

                                extra={

                                    "risk_reasons":
                                        risk_reasons,

                                    "risk_reason_text":
                                        reason_text
                                }
                            )
                        )

                        events.append(
                            risk_event
                        )

            # ======================================================
            # VEHICLE ANALYTICS / ANPR
            # ======================================================

            if label in self.VEHICLE_CLASSES:

                if track_id >= 0:

                    vehicle_key = (
                        str(camera_id),
                        int(track_id)
                    )

                    now = time.time()

                    last_check = (
                        self.last_plate_check.get(
                            vehicle_key,
                            0
                        )
                    )

                    # OCR at most every 2 seconds
                    # for a tracked vehicle.

                    if (
                        now - last_check
                        >= 2.0
                    ):

                        self.last_plate_check[
                            vehicle_key
                        ] = now

                        vehicle_roi = frame[
                            y1:y2,
                            x1:x2
                        ]

                        if vehicle_roi.size > 0:

                            try:

                                (
                                    plate_text,
                                    plate_conf
                                ) = self.anpr.read_plate(
                                    vehicle_roi
                                )

                            except Exception as error:

                                print(
                                    f"[ANPR] Error for "
                                    f"Track {track_id}: "
                                    f"{error}"
                                )

                                plate_text = None
                                plate_conf = 0.0

                            if plate_text:

                                self.plate_cache[
                                    vehicle_key
                                ] = (
                                    plate_text,
                                    plate_conf
                                )

                                anpr_event = (
                                    self._build_event(

                                        camera_id=camera_id,

                                        event_type=
                                            "ANPR_DETECTION",

                                        details=(
                                            f"Vehicle ID "
                                            f"{track_id} "
                                            f"plate: "
                                            f"{plate_text}"
                                        ),

                                        track_id=track_id,

                                        confidence=plate_conf,

                                        behavior=behavior,

                                        extra={

                                            "plate_text":
                                                plate_text,

                                            "plate_confidence":
                                                plate_conf,

                                            "vehicle_class":
                                                label
                                        }
                                    )
                                )

                                events.append(
                                    anpr_event
                                )

                    # ------------------------------------------------
                    # Use cached plate result
                    # ------------------------------------------------

                    if (
                        not plate_text
                        and
                        vehicle_key
                        in self.plate_cache
                    ):

                        (
                            plate_text,
                            plate_conf
                        ) = self.plate_cache[
                            vehicle_key
                        ]

            # ======================================================
            # DETECTION OUTPUT
            # ======================================================

            detections.append({

                "bbox":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                "label":
                    label,

                "conf":
                    conf,

                "track_id":
                    track_id,

                "camera_id":
                    camera_id,

                "center":
                    center,

                "plate_text":
                    plate_text,

                "plate_conf":
                    plate_conf,

                "event":
                    event,

                # ==================================================
                # BEHAVIOR / RISK INFORMATION
                # ==================================================

                "zone":
                    behavior.get(
                        "zone",
                        "UNKNOWN"
                    ),

                "direction":
                    behavior.get(
                        "direction",
                        "UNKNOWN"
                    ),

                "distance_to_perimeter":
                    behavior.get(
                        "distance_to_perimeter"
                    ),

                "dwell_time":
                    behavior.get(
                        "dwell_time",
                        0.0
                    ),

                "approaching":
                    behavior.get(
                        "approaching",
                        False
                    ),

                "behavior_loitering":
                    behavior.get(
                        "loitering",
                        False
                    ),

                "risk_score":
                    behavior.get(
                        "risk_score",
                        0
                    ),

                "risk_level":
                    behavior.get(
                        "risk_level",
                        "LOW"
                    ),

                "risk_trend":
                    behavior.get(
                        "risk_trend",
                        "STABLE"
                    ),

                "risk_escalating":
                    behavior.get(
                        "risk_escalating",
                        False
                    ),

                "risk_reasons":
                    behavior.get(
                        "risk_reasons",
                        []
                    ),

                # ==================================================
                # EARLY-WARNING INTELLIGENCE
                # ==================================================

                "early_warning":
                    behavior.get(
                        "early_warning",
                        False
                    ),

                "early_warning_reasons":
                    behavior.get(
                        "early_warning_reasons",
                        []
                    ),

                "approach_count":
                    behavior.get(
                        "approach_count",
                        0
                    ),

                "approach_duration":
                    behavior.get(
                        "approach_duration",
                        0.0
                    ),

                "risk_increase_streak":
                    behavior.get(
                        "risk_increase_streak",
                        0
                    ),

                # ==================================================
                # TRACK LIFECYCLE INFORMATION
                # ==================================================

                "first_seen":
                    lifecycle[
                        "first_seen"
                    ]
                    if lifecycle
                    else None,

                "last_seen":
                    lifecycle[
                        "last_seen"
                    ]
                    if lifecycle
                    else None,

                "track_duration":
                    lifecycle[
                        "duration"
                    ]
                    if lifecycle
                    else 0.0,

                "detection_count":
                    lifecycle[
                        "detection_count"
                    ]
                    if lifecycle
                    else 0,

                "average_confidence":
                    lifecycle[
                        "average_confidence"
                    ]
                    if lifecycle
                    else 0.0,

                # ==================================================
                # SPATIAL INTELLIGENCE
                # ==================================================

                "spatial_zone_id":
                    spatial.get(
                        "zone_id",
                        "OUTSIDE_DEFINED_ZONES"
                    ),

                "spatial_zone_name":
                    spatial.get(
                        "zone_name",
                        "Outside Defined Zones"
                    ),

                "spatial_zone_type":
                    spatial.get(
                        "zone_type",
                        "UNKNOWN"
                    ),

                "spatial_zone_dwell_time":
                    spatial.get(
                        "dwell_time",
                        0.0
                    ),

                "spatial_zone_changed":
                    spatial.get(
                        "zone_changed",
                        False
                    ),

                "previous_spatial_zone":
                    spatial.get(
                        "previous_zone_id"
                    ),

                # ==================================================
                # CROSS-CAMERA ENTITY CORRELATION
                # ==================================================

                "appearance_feature":
                    appearance_feature
            })

        # ==========================================================
        # CLEANUP
        # ==========================================================

        self._cleanup_tracks(
            camera_id,
            active_track_ids
        )

        return detections, events