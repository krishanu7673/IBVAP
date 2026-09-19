import time
import cv2


class CameraHealthMonitor:

    def __init__(
        self,
        freeze_threshold=5.0,
        tamper_threshold=0.35,
        dark_threshold=25.0,
        bright_threshold=235.0
    ):
        self.freeze_threshold = float(
            freeze_threshold
        )

        self.tamper_threshold = float(
            tamper_threshold
        )

        self.dark_threshold = float(
            dark_threshold
        )

        self.bright_threshold = float(
            bright_threshold
        )

        self.camera_states = {}

    # ========================================================
    # CREATE SMALL FRAME REPRESENTATION
    # ========================================================

    def _prepare_frame(self, frame):

        if frame is None:
            return None

        try:

            small = cv2.resize(
                frame,
                (160, 90),
                interpolation=cv2.INTER_AREA
            )

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY
            )

            return gray

        except Exception:

            return None

    # ========================================================
    # FRAME QUALITY ANALYSIS
    # ========================================================

    def _analyze_frame_quality(
        self,
        comparison_frame
    ):

        if comparison_frame is None:

            return {
                "brightness": 0.0,
                "contrast": 0.0,
                "edge_density": 0.0,
                "dark_frame": False,
                "bright_frame": False,
                "low_detail": False
            }

        try:

            brightness = float(
                comparison_frame.mean()
            )

            contrast = float(
                comparison_frame.std()
            )

            # ------------------------------------------------
            # Edge detection
            # ------------------------------------------------

            edges = cv2.Canny(
                comparison_frame,
                50,
                150
            )

            edge_density = float(
                cv2.countNonZero(edges)
                / edges.size
            )

            dark_frame = (
                brightness
                <= self.dark_threshold
            )

            bright_frame = (
                brightness
                >= self.bright_threshold
            )

            # Very low contrast + very low edge density
            # can indicate an obstructed/covered camera.

            low_detail = (
                contrast < 8.0
                and edge_density < 0.01
            )

            return {
                "brightness": round(
                    brightness,
                    2
                ),
                "contrast": round(
                    contrast,
                    2
                ),
                "edge_density": round(
                    edge_density,
                    4
                ),
                "dark_frame": dark_frame,
                "bright_frame": bright_frame,
                "low_detail": low_detail
            }

        except Exception:

            return {
                "brightness": 0.0,
                "contrast": 0.0,
                "edge_density": 0.0,
                "dark_frame": False,
                "bright_frame": False,
                "low_detail": False
            }

    # ========================================================
    # CAMERA TAMPER DETECTION
    # ========================================================

    def _detect_tampering(
        self,
        state,
        current_frame
    ):

        quality = self._analyze_frame_quality(
            current_frame
        )

        brightness = quality[
            "brightness"
        ]

        contrast = quality[
            "contrast"
        ]

        edge_density = quality[
            "edge_density"
        ]

        dark_frame = quality[
            "dark_frame"
        ]

        bright_frame = quality[
            "bright_frame"
        ]

        low_detail = quality[
            "low_detail"
        ]

        # ----------------------------------------------------
        # Suspicious visual conditions
        # ----------------------------------------------------

        suspicious = False

        tamper_reason = "NONE"

        # Completely / almost completely dark scene
        if dark_frame:

            suspicious = True

            tamper_reason = (
                "DARK_FRAME"
            )

        # Almost completely overexposed scene
        elif bright_frame:

            suspicious = True

            tamper_reason = (
                "OVEREXPOSED_FRAME"
            )

        # Very low-detail scene
        elif low_detail:

            suspicious = True

            tamper_reason = (
                "LOW_DETAIL"
            )

        # ----------------------------------------------------
        # Track duration of suspicious condition
        # ----------------------------------------------------

        if suspicious:

            if state.get(
                "tamper_since"
            ) is None:

                state[
                    "tamper_since"
                ] = time.time()

        else:

            state[
                "tamper_since"
            ] = None

            state[
                "tamper_reason"
            ] = "NONE"

        tamper_duration = 0.0

        if state.get(
            "tamper_since"
        ) is not None:

            tamper_duration = (
                time.time()
                - state[
                    "tamper_since"
                ]
            )

        # ----------------------------------------------------
        # Require persistence before declaring tampering.
        # This avoids triggering from a single dark frame.
        # ----------------------------------------------------

        tamper_detected = (
            suspicious
            and tamper_duration
            >= self.tamper_threshold
        )

        if tamper_detected:

            state[
                "tamper_reason"
            ] = tamper_reason

        return {
            "tamper_detected":
                tamper_detected,

            "tamper_reason":
                state.get(
                    "tamper_reason",
                    "NONE"
                ),

            "tamper_duration":
                round(
                    tamper_duration,
                    2
                ),

            "brightness":
                brightness,

            "contrast":
                contrast,

            "edge_density":
                edge_density
        }

    # ========================================================
    # UPDATE CAMERA STATE
    # ========================================================

    def update(
        self,
        camera_id,
        frame,
        frame_timestamp=None
    ):

        current_time = time.time()

        state = self.camera_states.setdefault(
            camera_id,
            {
                "last_frame_time": 0.0,
                "last_frame": None,
                "comparison_frame": None,
                "frame_count": 0,
                "start_time": current_time,
                "status": "UNKNOWN",
                "fps": 0.0,
                "unchanged_since": None,

                # Tamper state
                "tamper_since": None,
                "tamper_reason": "NONE",
                "tamper_detected": False,

                # Frame quality
                "brightness": 0.0,
                "contrast": 0.0,
                "edge_density": 0.0,

                # Recovery tracking
                "last_status_change": current_time
            }
        )

        # ====================================================
        # CAMERA OFFLINE
        # ====================================================

        if frame is None:

            previous_status = state.get(
                "status",
                "UNKNOWN"
            )

            state[
                "status"
            ] = "OFFLINE"

            state[
                "fps"
            ] = 0.0

            state[
                "tamper_detected"
            ] = False

            state[
                "tamper_reason"
            ] = "NONE"

            if previous_status != "OFFLINE":

                state[
                    "last_status_change"
                ] = current_time

            return self._build_status(
                camera_id,
                state
            )

        # ====================================================
        # FRAME TIMESTAMP
        # ====================================================

        if frame_timestamp is None:

            frame_timestamp = current_time

        state[
            "last_frame_time"
        ] = frame_timestamp

        state[
            "frame_count"
        ] += 1

        # ====================================================
        # FPS CALCULATION
        # ====================================================

        elapsed = (
            current_time
            - state[
                "start_time"
            ]
        )

        if elapsed > 0:

            state[
                "fps"
            ] = (
                state[
                    "frame_count"
                ]
                / elapsed
            )

        # ====================================================
        # PREPARE FRAME
        # ====================================================

        comparison_frame = (
            self._prepare_frame(
                frame
            )
        )

        # ====================================================
        # FREEZE DETECTION
        # ====================================================

        frozen = False

        previous_comparison = (
            state[
                "comparison_frame"
            ]
        )

        if (
            comparison_frame is not None
            and previous_comparison is not None
        ):

            try:

                difference = cv2.absdiff(
                    previous_comparison,
                    comparison_frame
                )

                mean_difference = float(
                    difference.mean()
                )

                # ------------------------------------------------
                # Almost no pixel change
                # ------------------------------------------------

                if mean_difference < 0.5:

                    if (
                        state[
                            "unchanged_since"
                        ]
                        is None
                    ):

                        state[
                            "unchanged_since"
                        ] = current_time

                    unchanged_duration = (
                        current_time
                        - state[
                            "unchanged_since"
                        ]
                    )

                    if (
                        unchanged_duration
                        >= self.freeze_threshold
                    ):

                        frozen = True

                else:

                    state[
                        "unchanged_since"
                    ] = None

            except Exception:

                state[
                    "unchanged_since"
                ] = None

        else:

            state[
                "unchanged_since"
            ] = None

        # ====================================================
        # STORE COMPARISON FRAME
        # ====================================================

        state[
            "comparison_frame"
        ] = comparison_frame

        # ====================================================
        # FRAME QUALITY / TAMPER ANALYSIS
        # ====================================================

        tamper_result = (
            self._detect_tampering(
                state,
                comparison_frame
            )
        )

        state[
            "tamper_detected"
        ] = tamper_result[
            "tamper_detected"
        ]

        state[
            "tamper_reason"
        ] = tamper_result[
            "tamper_reason"
        ]

        state[
            "brightness"
        ] = tamper_result[
            "brightness"
        ]

        state[
            "contrast"
        ] = tamper_result[
            "contrast"
        ]

        state[
            "edge_density"
        ] = tamper_result[
            "edge_density"
        ]

        # ====================================================
        # STALE FRAME DETECTION
        # ====================================================

        stale = (
            current_time
            - frame_timestamp
            > self.freeze_threshold
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        previous_status = state.get(
            "status",
            "UNKNOWN"
        )

        if stale:

            new_status = "STALE"

        elif frozen:

            new_status = "FROZEN"

        elif state[
            "tamper_detected"
        ]:

            new_status = "TAMPERED"

        else:

            new_status = "ONLINE"

        state[
            "status"
        ] = new_status

        # ----------------------------------------------------
        # Track status changes
        # ----------------------------------------------------

        if new_status != previous_status:

            state[
                "last_status_change"
            ] = current_time

        # ====================================================
        # RETURN
        # ====================================================

        return self._build_status(
            camera_id,
            state
        )

    # ========================================================
    # BUILD STATUS RESPONSE
    # ========================================================

    def _build_status(
        self,
        camera_id,
        state
    ):

        return {
            "camera_id":
                camera_id,

            "status":
                state.get(
                    "status",
                    "UNKNOWN"
                ),

            "fps":
                round(
                    state.get(
                        "fps",
                        0.0
                    ),
                    2
                ),

            "tamper_detected":
                state.get(
                    "tamper_detected",
                    False
                ),

            "tamper_reason":
                state.get(
                    "tamper_reason",
                    "NONE"
                ),

            "tamper_duration":
                round(
                    (
                        time.time()
                        - state[
                            "tamper_since"
                        ]
                    )
                    if state.get(
                        "tamper_since"
                    ) is not None
                    else 0.0,
                    2
                ),

            "brightness":
                round(
                    state.get(
                        "brightness",
                        0.0
                    ),
                    2
                ),

            "contrast":
                round(
                    state.get(
                        "contrast",
                        0.0
                    ),
                    2
                ),

            "edge_density":
                round(
                    state.get(
                        "edge_density",
                        0.0
                    ),
                    4
                ),

            "last_status_change":
                state.get(
                    "last_status_change"
                )
        }

    # ========================================================
    # GET SINGLE CAMERA STATUS
    # ========================================================

    def get_status(
        self,
        camera_id
    ):

        state = self.camera_states.get(
            camera_id
        )

        if state is None:

            return {
                "camera_id":
                    camera_id,

                "status":
                    "UNKNOWN",

                "fps":
                    0.0,

                "tamper_detected":
                    False,

                "tamper_reason":
                    "NONE",

                "tamper_duration":
                    0.0,

                "brightness":
                    0.0,

                "contrast":
                    0.0,

                "edge_density":
                    0.0
            }

        return self._build_status(
            camera_id,
            state
        )

    # ========================================================
    # GET ALL CAMERA STATUS
    # ========================================================

    def get_all_status(self):

        result = {}

        for camera_id in (
            self.camera_states
        ):

            result[
                camera_id
            ] = self.get_status(
                camera_id
            )

        return result

    # ========================================================
    # GET HEALTH STATISTICS
    # ========================================================

    def get_statistics(self):

        total = len(
            self.camera_states
        )

        online = 0
        offline = 0
        frozen = 0
        stale = 0
        tampered = 0

        for state in (
            self.camera_states.values()
        ):

            status = state.get(
                "status",
                "UNKNOWN"
            )

            if status == "ONLINE":
                online += 1

            elif status == "OFFLINE":
                offline += 1

            elif status == "FROZEN":
                frozen += 1

            elif status == "STALE":
                stale += 1

            elif status == "TAMPERED":
                tampered += 1

        return {
            "total_cameras":
                total,

            "online":
                online,

            "offline":
                offline,

            "frozen":
                frozen,

            "stale":
                stale,

            "tampered":
                tampered
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
        max_age=60
    ):

        current_time = time.time()

        expired = []

        for (
            camera_id,
            state
        ) in self.camera_states.items():

            last_frame_time = state.get(
                "last_frame_time",
                0
            )

            if (
                last_frame_time > 0
                and
                current_time
                - last_frame_time
                > max_age
            ):

                expired.append(
                    camera_id
                )

        for camera_id in expired:

            del self.camera_states[
                camera_id
            ]