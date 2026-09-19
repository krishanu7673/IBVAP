import os
import json
import time

# ============================================================
# GPU / ONNX RUNTIME INITIALIZATION
# ============================================================

import torch
import onnxruntime as ort
from src.incident_correlator import IncidentCorrelator


def prepare_gpu_environment():
    print("=" * 60)
    print("IBVAP GPU ENVIRONMENT")
    print("=" * 60)

    print(
        "Python:",
        __import__("sys").version.split()[0]
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "ONNX Runtime:",
        ort.__version__
    )

    print(
        "ORT providers:",
        ort.get_available_providers()
    )

    # --------------------------------------------------------
    # Windows:
    # Allow ONNX Runtime to find PyTorch CUDA/cuDNN DLLs.
    # --------------------------------------------------------

    torch_lib = os.path.join(
        os.path.dirname(torch.__file__),
        "lib"
    )

    if os.path.isdir(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)

        print(
            "PyTorch DLL directory:",
            torch_lib
        )

    # ORT >= 1.21
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()

            print(
                "[OK] ONNX Runtime CUDA DLLs preloaded."
            )

        except Exception as exc:
            print(
                "[WARNING] ORT DLL preload failed:",
                exc
            )

    print("=" * 60)


# ============================================================
# IMPORT PROJECT MODULES AFTER GPU SETUP
# ============================================================

from src.ingestion import ThreadedRTSPStream
from src.analytics import BorderAnalyticsEngine
from src.events import EventManager
from src.storage import SecureStorageBuffer
from src.alert_manager import AlertManager
from src.evidence import EvidenceManager
from src.system_health import CameraHealthMonitor


# ============================================================
# CROSS-CAMERA / RE-ID TEMPORARILY DISABLED
# ============================================================

# Step 1H remains preserved but disabled.

# from src.cross_camera import CrossCameraCorrelator


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_FILE = "config/cameras.json"
MODEL_PATH = "yolov8n.onnx"
DATABASE_PATH = "ibvap_border_events.db"


# ============================================================
# LOAD CAMERA CONFIGURATION
# ============================================================

def load_camera_config():

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Camera configuration not found: {CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    cameras = config.get("cameras", [])

    if not cameras:
        raise ValueError(
            "No cameras configured in cameras.json"
        )

    return cameras


# ============================================================
# DRAW TEXT HELPER
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.50,
    thickness=1
):

    import cv2

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(
    frame,
    camera_id,
    detections,
    perimeter_line,
    health_status=None,
    performance=None
):

    import cv2

    frame_height, frame_width = frame.shape[:2]

    # ========================================================
    # VIRTUAL PERIMETER
    # ========================================================

    cv2.line(
        frame,
        tuple(perimeter_line[0]),
        tuple(perimeter_line[1]),
        (0, 0, 255),
        3
    )

    draw_text(
        frame,
        "VIRTUAL PERIMETER",
        (
            perimeter_line[0][0],
            max(20, perimeter_line[0][1] - 12)
        ),
        (0, 0, 255),
        0.65,
        2
    )

    # ========================================================
    # CAMERA HEALTH DATA
    # ========================================================

    health_status = health_status or {}

    health_state = health_status.get(
        "status",
        "UNKNOWN"
    )

    capture_fps = health_status.get(
        "fps",
        0.0
    )

    # ========================================================
    # SYSTEM HEADER
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (610, 78),
        (20, 20, 20),
        -1
    )

    if health_state == "ONLINE":
        status_color = (0, 255, 0)

    elif health_state == "FROZEN":
        status_color = (0, 165, 255)

    elif health_state == "STALE":
        status_color = (0, 255, 255)

    elif health_state == "OFFLINE":
        status_color = (0, 0, 255)

    else:
        status_color = (180, 180, 180)

    draw_text(
        frame,
        f"CAMERA HEALTH: {health_state}",
        (20, 38),
        status_color,
        0.55,
        2
    )

    draw_text(
        frame,
        f"CAMERA: {camera_id}",
        (300, 38),
        (255, 255, 255),
        0.48,
        2
    )

    draw_text(
        frame,
        f"CAPTURE FPS: {capture_fps:.2f}",
        (20, 65),
        (255, 255, 255),
        0.46,
        1
    )

    draw_text(
        frame,
        "IBVAP INTELLIGENCE ENGINE",
        (190, 65),
        (255, 255, 255),
        0.43,
        1
    )

    # ========================================================
    # PERFORMANCE PANEL
    # ========================================================

    if performance is not None:

        frames_processed = performance.get(
            "frames_processed",
            0
        )

        processing_fps = performance.get(
            "fps",
            0
        )

        last_processing_ms = performance.get(
            "last_processing_time_ms",
            0
        )

        average_processing_ms = performance.get(
            "average_processing_time_ms",
            0
        )

        cv2.rectangle(
            frame,
            (
                max(0, frame_width - 360),
                10
            ),
            (
                frame_width - 10,
                105
            ),
            (20, 20, 20),
            -1
        )

        draw_text(
            frame,
            "SYSTEM PERFORMANCE",
            (
                frame_width - 345,
                32
            ),
            (255, 255, 255),
            0.52,
            2
        )

        draw_text(
            frame,
            f"AI FPS: {processing_fps:.1f}",
            (
                frame_width - 345,
                55
            ),
            (0, 255, 0),
            0.48,
            1
        )

        draw_text(
            frame,
            f"INFERENCE: {last_processing_ms:.1f} ms",
            (
                frame_width - 200,
                55
            ),
            (255, 255, 255),
            0.42,
            1
        )

        draw_text(
            frame,
            f"AVG: {average_processing_ms:.1f} ms",
            (
                frame_width - 345,
                80
            ),
            (255, 255, 255),
            0.45,
            1
        )

        draw_text(
            frame,
            f"FRAMES: {frames_processed}",
            (
                frame_width - 200,
                80
            ),
            (255, 255, 255),
            0.45,
            1
        )

    # ========================================================
    # GLOBAL TRACK / RISK SUMMARY
    # ========================================================

    tracked_count = 0
    medium_count = 0
    high_count = 0
    critical_count = 0

    for det in detections:

        track_id = det.get(
            "track_id",
            -1
        )

        if track_id >= 0:
            tracked_count += 1

        risk_level = det.get(
            "risk_level",
            "LOW"
        )

        if risk_level == "MEDIUM":
            medium_count += 1

        elif risk_level == "HIGH":
            high_count += 1

        elif risk_level == "CRITICAL":
            critical_count += 1

    cv2.rectangle(
        frame,
        (10, 90),
        (300, 165),
        (20, 20, 20),
        -1
    )

    draw_text(
        frame,
        f"TRACKED OBJECTS: {tracked_count}",
        (20, 112),
        (255, 255, 255),
        0.48,
        1
    )

    draw_text(
        frame,
        f"MEDIUM RISK: {medium_count}",
        (20, 132),
        (255, 255, 255),
        0.43,
        1
    )

    draw_text(
        frame,
        f"HIGH RISK: {high_count}",
        (20, 150),
        (0, 165, 255),
        0.43,
        1
    )

    draw_text(
        frame,
        f"CRITICAL: {critical_count}",
        (155, 150),
        (0, 0, 255),
        0.43,
        1
    )

    # ========================================================
    # DETECTION BOXES
    # ========================================================

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        label = det["label"]

        confidence = det["conf"]

        track_id = det["track_id"]

        plate_text = det["plate_text"]

        risk_score = det.get(
            "risk_score",
            0
        )

        risk_level = det.get(
            "risk_level",
            "LOW"
        )

        zone = det.get(
            "zone",
            "UNKNOWN"
        )

        direction = det.get(
            "direction",
            "UNKNOWN"
        )

        dwell_time = det.get(
            "dwell_time",
            0.0
        )

        track_duration = det.get(
            "track_duration",
            0.0
        )

        detection_count = det.get(
            "detection_count",
            0
        )

        average_confidence = det.get(
            "average_confidence",
            0.0
        )

        risk_trend = det.get(
            "risk_trend",
            "STABLE"
        )

        risk_escalating = det.get(
            "risk_escalating",
            False
        )

        risk_reasons = det.get(
            "risk_reasons",
            []
        )

        # ====================================================
        # BOX COLOR
        # ====================================================

        if label == "person":
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)

        if risk_level == "HIGH":
            color = (0, 165, 255)

        elif risk_level == "CRITICAL":
            color = (0, 0, 255)

        if det.get("event"):
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ====================================================
        # MAIN OBJECT LABEL
        # ====================================================

        main_label = (
            f"ID:{track_id} "
            f"{label.upper()} "
            f"{confidence:.2f}"
        )

        draw_text(
            frame,
            main_label,
            (
                x1,
                max(20, y1 - 10)
            ),
            color,
            0.50,
            2
        )

        # ====================================================
        # INTELLIGENCE PANEL
        # ====================================================

        if label == "person":

            panel_width = 330
            panel_height = 220

            panel_x = x1
            panel_y = y2 + 5

            if (
                panel_y + panel_height
                >= frame_height
            ):
                panel_y = max(
                    5,
                    y1 - panel_height - 5
                )

            if (
                panel_x + panel_width
                >= frame_width
            ):
                panel_x = max(
                    5,
                    frame_width
                    - panel_width
                    - 5
                )

            cv2.rectangle(
                frame,
                (
                    panel_x,
                    panel_y
                ),
                (
                    panel_x + panel_width,
                    panel_y + panel_height
                ),
                (20, 20, 20),
                -1
            )

            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            if risk_level == "CRITICAL":
                risk_color = (0, 0, 255)

            elif risk_level == "HIGH":
                risk_color = (0, 165, 255)

            elif risk_level == "MEDIUM":
                risk_color = (0, 255, 255)

            else:
                risk_color = (0, 255, 0)

            draw_text(
                frame,
                (
                    f"RISK: "
                    f"{risk_score}/100 "
                    f"{risk_level}"
                ),
                (
                    panel_x + 8,
                    panel_y + 20
                ),
                risk_color,
                0.50,
                2
            )

            # ------------------------------------------------
            # RISK TREND
            # ------------------------------------------------

            trend_symbol = "->"

            if risk_trend == "INCREASING":
                trend_symbol = "^"

            elif risk_trend == "DECREASING":
                trend_symbol = "v"

            trend_text = (
                f"TREND: "
                f"{trend_symbol} "
                f"{risk_trend}"
            )

            if risk_escalating:
                trend_text += " ESCALATING"

            draw_text(
                frame,
                trend_text,
                (
                    panel_x + 8,
                    panel_y + 40
                ),
                (
                    (0, 165, 255)
                    if risk_escalating
                    else (255, 255, 255)
                ),
                0.42,
                1
            )

            # ------------------------------------------------
            # BEHAVIOR REASONS
            # ------------------------------------------------

            draw_text(
                frame,
                "RISK FACTORS:",
                (
                    panel_x + 8,
                    panel_y + 60
                ),
                (255, 255, 255),
                0.40,
                1
            )

            displayed_reasons = []

            if isinstance(
                risk_reasons,
                list
            ):
                displayed_reasons = (
                    risk_reasons[:3]
                )

            for reason_index, reason in enumerate(
                displayed_reasons
            ):

                reason_text = str(reason)

                if len(reason_text) > 42:
                    reason_text = (
                        reason_text[:39]
                        + "..."
                    )

                draw_text(
                    frame,
                    f"- {reason_text}",
                    (
                        panel_x + 8,
                        panel_y + 78
                        + reason_index * 17
                    ),
                    (220, 220, 220),
                    0.36,
                    1
                )

            # ------------------------------------------------
            # ZONE
            # ------------------------------------------------

            draw_text(
                frame,
                f"ZONE: {zone}",
                (
                    panel_x + 8,
                    panel_y + 132
                ),
                (255, 255, 255),
                0.42,
                1
            )

            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            draw_text(
                frame,
                f"DIRECTION: {direction}",
                (
                    panel_x + 8,
                    panel_y + 151
                ),
                (255, 255, 255),
                0.42,
                1
            )

            # ------------------------------------------------
            # DWELL
            # ------------------------------------------------

            draw_text(
                frame,
                f"DWELL: {dwell_time:.1f}s",
                (
                    panel_x + 8,
                    panel_y + 170
                ),
                (255, 255, 255),
                0.42,
                1
            )

            # ------------------------------------------------
            # TRACK LIFECYCLE
            # ------------------------------------------------

            draw_text(
                frame,
                f"TRACK: {track_duration:.1f}s",
                (
                    panel_x + 8,
                    panel_y + 189
                ),
                (255, 255, 255),
                0.42,
                1
            )

            # ------------------------------------------------
            # OBSERVATIONS
            # ------------------------------------------------

            draw_text(
                frame,
                (
                    f"OBS: {detection_count} "
                    f"AVG CONF: "
                    f"{average_confidence:.2f}"
                ),
                (
                    panel_x + 8,
                    panel_y + 207
                ),
                (200, 200, 200),
                0.37,
                1
            )

        # ====================================================
        # LICENSE PLATE
        # ====================================================

        if plate_text:

            plate_label = (
                f"PLATE: {plate_text}"
            )

            plate_y = min(
                frame_height - 10,
                y2 + 20
            )

            draw_text(
                frame,
                plate_label,
                (
                    x1,
                    plate_y
                ),
                (0, 165, 255),
                0.55,
                2
            )


# ============================================================
# LEGACY INCIDENT ID GENERATOR
# ============================================================

def generate_incident_id(
    camera_id,
    track_id
):

    safe_camera = (
        str(camera_id)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    if track_id is None:
        track_part = "GLOBAL"
    else:
        track_part = str(track_id)

    return (
        f"INC_{safe_camera}_"
        f"{track_part}"
    )


# ============================================================
# ALERT PRIORITY
# ============================================================

def get_alert_priority(event):

    risk_score = event.get(
        "risk_score"
    )

    risk_level = event.get(
        "risk_level"
    )

    event_type = event.get(
        "type",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # Explicit critical events
    # --------------------------------------------------------

    if event_type == "PERIMETER_BREACH":
        return "CRITICAL"

    # --------------------------------------------------------
    # Risk-based priority
    # --------------------------------------------------------

    if risk_level == "CRITICAL":
        return "CRITICAL"

    if risk_level == "HIGH":
        return "HIGH"

    if risk_score is not None:

        try:

            score = float(
                risk_score
            )

            if score >= 75:
                return "CRITICAL"

            if score >= 50:
                return "HIGH"

            if score >= 25:
                return "MEDIUM"

        except Exception:
            pass

    # --------------------------------------------------------
    # Event-specific priority
    # --------------------------------------------------------

    if event_type == "LOITERING_DETECTED":
        return "MEDIUM"

    if event_type == "ANPR_DETECTION":
        return "LOW"

    return "LOW"


# ============================================================
# MAIN
# ============================================================

def main():

    import cv2

    prepare_gpu_environment()

    print(
        "\n[*] Loading camera configuration..."
    )

    cameras = load_camera_config()

    print(
        f"[*] Cameras configured: {len(cameras)}"
    )

    # --------------------------------------------------------
    # Secure storage
    # --------------------------------------------------------

    storage = SecureStorageBuffer(
        DATABASE_PATH
    )

    # --------------------------------------------------------
    # Event manager
    # --------------------------------------------------------

    event_manager = EventManager(
        default_cooldown=5.0
    )

    # --------------------------------------------------------
    # Intelligent alert management
    # --------------------------------------------------------

    alert_manager = AlertManager(
        cooldown=10
    )

    # --------------------------------------------------------
    # Automatic evidence management
    # --------------------------------------------------------

    evidence_manager = EvidenceManager(
        base_directory="evidence"
    )

    # --------------------------------------------------------
    # Camera health and tamper monitoring
    # --------------------------------------------------------

    health_monitor = CameraHealthMonitor(
        freeze_threshold=5.0
    )

    # ========================================================
    # INCIDENT CORRELATION
    # ========================================================

    incident_correlator = IncidentCorrelator(
        correlation_window=60.0,
        incident_timeout=120.0
    )

    print(
        "[OK] Intelligent incident correlation: ACTIVE."
    )

    # ========================================================
    # CROSS-CAMERA CORRELATION TEMPORARILY DISABLED
    # ========================================================

    # Step 1H remains intentionally disabled.

    # entity_correlator = CrossCameraCorrelator(
    #     similarity_threshold=0.80,
    #     max_entity_age=60.0
    # )

    streams = {}
    analytics_engines = {}

    # ========================================================
    # Initialize cameras
    # ========================================================

    for camera in cameras:

        if not camera.get(
            "enabled",
            True
        ):
            continue

        camera_id = camera["id"]
        source = camera["source"]

        # ----------------------------------------------------
        # Perimeter configuration
        # ----------------------------------------------------

        perimeter_line = camera.get(
            "perimeter_line",
            [
                [50, 400],
                [600, 400]
            ]
        )

        # ----------------------------------------------------
        # Detection configuration
        # ----------------------------------------------------

        detection_config = camera.get(
            "detection",
            {}
        )

        confidence_threshold = float(
            detection_config.get(
                "confidence",
                0.35
            )
        )

        iou_threshold = float(
            detection_config.get(
                "iou",
                0.50
            )
        )

        # ----------------------------------------------------
        # Behavior configuration
        # ----------------------------------------------------

        behavior_config = camera.get(
            "behavior",
            {}
        )

        loitering_seconds = float(
            behavior_config.get(
                "loitering_seconds",
                20.0
            )
        )

        print(
            f"\n[*] Initializing {camera_id}"
        )

        print(
            f"    Source: {source}"
        )

        print(
            f"    Confidence threshold: "
            f"{confidence_threshold}"
        )

        print(
            f"    IoU threshold: "
            f"{iou_threshold}"
        )

        print(
            f"    Loitering threshold: "
            f"{loitering_seconds}s"
        )

        print(
            f"    Perimeter: "
            f"{perimeter_line}"
        )

        try:

            stream = (
                ThreadedRTSPStream(
                    source,
                    camera_id
                )
                .start()
            )

            streams[
                camera_id
            ] = stream

            # ------------------------------------------------
            # Spatial zone configuration
            # ------------------------------------------------

            zones = camera.get(
                "zones",
                []
            )

            # ------------------------------------------------
            # Separate analytics engine per camera
            # ------------------------------------------------

            analytics_engines[
                camera_id
            ] = BorderAnalyticsEngine(
                model_path=MODEL_PATH,
                use_gpu=True,
                perimeter_line=perimeter_line,
                loitering_seconds=loitering_seconds,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                zones=zones
            )

            print(
                f"[OK] {camera_id} online."
            )

        except Exception as exc:

            print(
                f"[ERROR] Could not initialize "
                f"{camera_id}: {exc}"
            )

    if not streams:

        print(
            "[FATAL] No cameras could be started."
        )

        return

    print(
        "\n[*] IBVAP PLATFORM ONLINE"
    )

    print(
        "[*] Press Q to quit."
    )

    print(
        "[*] Press I to verify database integrity."
    )

    print(
        "[*] Press E to show recent events."
    )

    print(
        "[*] Press S to show incident statistics."
    )

    print(
        "[*] HIGH/CRITICAL alerts automatically "
        "generate evidence snapshots."
    )

    print(
        "[*] Camera health monitoring: "
        "ONLINE / FROZEN / STALE / OFFLINE."
    )

    print(
        "[*] Cross-camera entity correlation: "
        "TEMPORARILY DISABLED."
    )

    print(
        "[*] Intelligent incident correlation: "
        "ACTIVE."
    )

    print("=" * 60)

    # ========================================================
    # Main loop
    # ========================================================

    try:

        while True:

            any_camera_running = False

            for camera in cameras:

                if not camera.get(
                    "enabled",
                    True
                ):
                    continue

                camera_id = camera["id"]

                if camera_id not in streams:
                    continue

                stream = streams[
                    camera_id
                ]

                analytics = (
                    analytics_engines[
                        camera_id
                    ]
                )

                perimeter_line = camera.get(
                    "perimeter_line",
                    [
                        [50, 400],
                        [600, 400]
                    ]
                )

                # ------------------------------------------------
                # Read frame
                # ------------------------------------------------

                status, frame = (
                    stream.read()
                )

                # ------------------------------------------------
                # Camera health when frame unavailable
                # ------------------------------------------------

                if (
                    not status
                    or frame is None
                ):

                    health_status = (
                        health_monitor.update(
                            camera_id,
                            None
                        )
                    )

                    continue

                # ------------------------------------------------
                # Use ingestion timestamp for stale detection
                # ------------------------------------------------

                frame_timestamp = getattr(
                    stream,
                    "last_frame_time",
                    None
                )

                health_status = (
                    health_monitor.update(
                        camera_id,
                        frame,
                        frame_timestamp
                    )
                )

                any_camera_running = True

                # =================================================
                # AI ANALYTICS
                # =================================================

                try:

                    (
                        detections,
                        events
                    ) = analytics.process_frame(
                        frame,
                        camera_id
                    )

                except Exception as exc:

                    print(
                        f"[AI ERROR] "
                        f"{camera_id}: {exc}"
                    )

                    continue

                # =================================================
                # PROCESS EVENTS
                # =================================================

                for event in events:

                    event_type = event[
                        "type"
                    ]

                    track_id = event.get(
                        "track_id"
                    )

                    details = event[
                        "details"
                    ]

                    confidence = event.get(
                        "confidence"
                    )

                    risk_score = event.get(
                        "risk_score"
                    )

                    zone = event.get(
                        "zone"
                    )

                    direction = event.get(
                        "direction"
                    )

                    # ------------------------------------------------
                    # Determine operational priority
                    # ------------------------------------------------

                    priority = get_alert_priority(
                        event
                    )

                    # ------------------------------------------------
                    # EventManager:
                    # Prevent duplicate event generation.
                    # ------------------------------------------------

                    final_event = (
                        event_manager.create_event(
                            camera_id=camera_id,
                            event_type=event_type,
                            details=details,
                            track_id=track_id,
                            confidence=confidence
                        )
                    )

                    if final_event is None:
                        continue

                    # =================================================
                    # INCIDENT CORRELATION
                    # =================================================

                    try:

                        correlation_event = dict(
                            event
                        )

                        correlation_event[
                            "camera_id"
                        ] = camera_id

                        correlation_event[
                            "timestamp"
                        ] = time.time()

                        correlation_event[
                            "event_type"
                        ] = event_type

                        correlation_event[
                            "track_id"
                        ] = track_id

                        correlation_event[
                            "details"
                        ] = details

                        correlation_event[
                            "confidence"
                        ] = confidence

                        correlation_event[
                            "risk_score"
                        ] = risk_score

                        correlation_event[
                            "zone"
                        ] = zone

                        correlation_event[
                            "direction"
                        ] = direction

                        incident = (
                            incident_correlator.add_event(
                                correlation_event
                            )
                        )

                    except Exception as exc:

                        incident = None

                        print(
                            f"[INCIDENT ERROR] "
                            f"{camera_id}: {exc}"
                        )

                    # ------------------------------------------------
                    # Incident ID
                    # ------------------------------------------------

                    if incident is not None:

                        incident_id = incident[
                            "incident_id"
                        ]

                    else:

                        incident_id = (
                            generate_incident_id(
                                camera_id,
                                track_id
                            )
                        )

                    # =================================================
                    # Automatic evidence capture
                    # =================================================

                    snapshot_path = None

                    if priority in (
                        "HIGH",
                        "CRITICAL"
                    ):

                        should_capture = (
                            alert_manager.should_alert(
                                camera_id,
                                track_id,
                                event_type
                            )
                        )

                        if should_capture:

                            try:

                                snapshot_path = (
                                    evidence_manager.save_snapshot(
                                        frame=frame,
                                        camera_id=camera_id,
                                        event_type=event_type,
                                        track_id=track_id
                                    )
                                )

                                if snapshot_path:

                                    print(
                                        "[EVIDENCE] "
                                        "Snapshot saved:"
                                    )

                                    print(
                                        f"    "
                                        f"{snapshot_path}"
                                    )

                                    # ------------------------------------------------
                                    # Attach evidence to correlated incident
                                    # ------------------------------------------------

                                    if incident is not None:

                                        incident_correlator.add_evidence(
                                            incident_id,
                                            snapshot_path
                                        )

                            except Exception as exc:

                                print(
                                    "[EVIDENCE ERROR]",
                                    exc
                                )

                    # =================================================
                    # Store event
                    # =================================================

                    storage.log_event(
                        camera_id=camera_id,
                        event_type=event_type,
                        details=details,
                        track_id=track_id,
                        confidence=confidence,
                        risk_score=risk_score,
                        zone=zone,
                        direction=direction,
                        snapshot_path=snapshot_path,
                        incident_id=incident_id
                    )

                    # =================================================
                    # INCIDENT STATUS OUTPUT
                    # =================================================

                    if incident is not None:

                        print(
                            "\n"
                            + "-" * 60
                        )

                        print(
                            "[INCIDENT CORRELATION]"
                        )

                        print(
                            f"Incident ID: "
                            f"{incident['incident_id']}"
                        )

                        print(
                            f"Entity ID: "
                            f"{incident.get('entity_id')}"
                        )

                        print(
                            f"Camera: "
                            f"{incident.get('camera_id')}"
                        )

                        print(
                            f"Event Count: "
                            f"{incident['event_count']}"
                        )

                        print(
                            f"Current Risk: "
                            f"{incident['current_risk']:.1f}/100"
                        )

                        print(
                            f"Peak Risk: "
                            f"{incident['peak_risk']:.1f}/100"
                        )

                        print(
                            f"Status: "
                            f"{incident['status']}"
                        )

                        print(
                            "Timeline Events: "
                            f"{', '.join(incident['event_types'])}"
                        )

                        print(
                            f"Evidence Items: "
                            f"{len(incident['evidence'])}"
                        )

                        print(
                            "-" * 60
                        )

                    # =================================================
                    # Console alert
                    # =================================================

                    print(
                        "\n"
                        + "=" * 60
                    )

                    print(
                        f"[{priority} ALERT]"
                    )

                    print(
                        f"Event: "
                        f"{event_type}"
                    )

                    print(
                        f"Camera: "
                        f"{camera_id}"
                    )

                    print(
                        f"Track ID: "
                        f"{track_id}"
                    )

                    print(
                        f"Incident ID: "
                        f"{incident_id}"
                    )

                    print(
                        f"Details: "
                        f"{details}"
                    )

                    if risk_score is not None:

                        print(
                            f"Risk Score: "
                            f"{risk_score}/100"
                        )

                    if zone is not None:

                        print(
                            f"Zone: "
                            f"{zone}"
                        )

                    if direction is not None:

                        print(
                            f"Direction: "
                            f"{direction}"
                        )

                    print(
                        f"Priority: "
                        f"{priority}"
                    )

                    if incident is not None:

                        print(
                            f"Incident Status: "
                            f"{incident['status']}"
                        )

                        print(
                            f"Peak Incident Risk: "
                            f"{incident['peak_risk']:.1f}/100"
                        )

                    if snapshot_path:

                        print(
                            f"Evidence: "
                            f"{snapshot_path}"
                        )

                    else:

                        print(
                            "Evidence: "
                            "Not captured"
                        )

                    print(
                        "=" * 60
                    )

                # =================================================
                # DISPLAY / HUD
                # =================================================

                display_frame = frame.copy()

                draw_hud(
                    display_frame,
                    camera_id,
                    detections,
                    perimeter_line,
                    health_status
                )

                # =================================================
                # OPEN VIDEO WINDOW
                # =================================================

                window_name = (
                    f"IBVAP - {camera_id}"
                )

                cv2.imshow(
                    window_name,
                    display_frame
                )

            # ====================================================
            # Incident cleanup
            # ====================================================

            incident_correlator.cleanup()

            # ====================================================
            # Keyboard
            # ====================================================

            key = cv2.waitKey(1) & 0xFF

            # ----------------------------------------------------
            # Q -> quit
            # ----------------------------------------------------

            if key == ord("q"):
                break

            # ----------------------------------------------------
            # I -> integrity verification
            # ----------------------------------------------------

            if key == ord("i"):

                valid, message = (
                    storage.verify_integrity()
                )

                print(
                    "\n[SECURITY INTEGRITY]"
                )

                if valid:

                    print(
                        "[OK]",
                        message
                    )

                else:

                    print(
                        "[CRITICAL]",
                        message
                    )

            # ----------------------------------------------------
            # E -> recent events
            # ----------------------------------------------------

            if key == ord("e"):

                print(
                    "\n[RECENT EVENTS]"
                )

                recent_events = (
                    storage.get_recent_events(
                        limit=10
                    )
                )

                for stored_event in recent_events:

                    print(
                        stored_event
                    )

            # ----------------------------------------------------
            # S -> incident statistics
            # ----------------------------------------------------

            if key == ord("s"):

                statistics = (
                    incident_correlator.get_statistics()
                )

                print(
                    "\n[INCIDENT STATISTICS]"
                )

                print(
                    f"Total incidents: "
                    f"{statistics['total_incidents']}"
                )

                print(
                    f"Active incidents: "
                    f"{statistics['active_incidents']}"
                )

                print(
                    f"Critical incidents: "
                    f"{statistics['critical_incidents']}"
                )

                print(
                    f"High incidents: "
                    f"{statistics['high_incidents']}"
                )

            # ====================================================
            # Check if all video files ended
            # ====================================================

            if not any_camera_running:

                all_files_finished = all(
                    (
                        not streams[
                            c["id"]
                        ].status
                        and isinstance(
                            c["source"],
                            str
                        )
                    )
                    for c in cameras
                    if (
                        c.get(
                            "enabled",
                            True
                        )
                        and c["id"] in streams
                    )
                )

                if all_files_finished:

                    print(
                        "\n[*] All video sources ended."
                    )

                    break

                time.sleep(
                    0.05
                )

    except KeyboardInterrupt:

        print(
            "\n[*] Keyboard interrupt."
        )

    finally:

        print(
            "\n[*] Shutting down..."
        )

        for stream in streams.values():

            stream.stop()

        cv2.destroyAllWindows()

        print(
            "[*] IBVAP shutdown complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()