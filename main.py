import os
import json
import time
import subprocess

# ============================================================
# GPU / ONNX RUNTIME INITIALIZATION
# ============================================================

import torch
import onnxruntime as ort

from src.incident_correlator import IncidentCorrelator
from src.incident_intelligence import IncidentIntelligence


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

            os.add_dll_directory(
                torch_lib
            )

        print(
            "PyTorch DLL directory:",
            torch_lib
        )

    # --------------------------------------------------------
    # ORT >= 1.21
    # --------------------------------------------------------

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
# CROSS-CAMERA ENTITY CORRELATION
# ============================================================

from src.cross_camera import CrossCameraCorrelator


# ============================================================
# OPTIONAL SYSTEM MONITORING
# ============================================================

try:

    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    psutil = None
    PSUTIL_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_FILE = "config/cameras.json"
MODEL_PATH = "yolov8n.onnx"
DATABASE_PATH = "ibvap_border_events.db"


# ============================================================
# SYSTEM PERFORMANCE MONITOR
# ============================================================

class SystemPerformanceMonitor:

    def __init__(
        self,
        metrics_interval=2.0,
        report_interval=10.0
    ):

        self.start_time = time.time()

        # ----------------------------------------------------
        # PERFORMANCE MONITORING INTERVALS
        #
        # System/GPU metrics are expensive, especially
        # nvidia-smi. They must NOT be queried every frame.
        # ----------------------------------------------------

        self.metrics_interval = float(
            metrics_interval
        )

        self.report_interval = float(
            report_interval
        )

        self.last_update_time = 0.0

        self.last_report_time = (
            self.start_time
        )

        # ----------------------------------------------------
        # GLOBAL COUNTERS
        # ----------------------------------------------------

        self.total_frames = 0

        self.total_unavailable_frames = 0

        self.total_processing_errors = 0

        # ----------------------------------------------------
        # CAMERA STATISTICS
        # ----------------------------------------------------

        self.camera_stats = {}

        # ----------------------------------------------------
        # SYSTEM METRICS
        # ----------------------------------------------------

        self.system_fps = 0.0

        self.cpu_percent = 0.0

        self.memory_percent = 0.0

        self.memory_used_gb = 0.0

        self.memory_total_gb = 0.0

        # ----------------------------------------------------
        # GPU METRICS
        # ----------------------------------------------------

        self.gpu_utilization = None

        self.gpu_memory_used_mb = None

        self.gpu_memory_total_mb = None

        self.gpu_temperature = None

        self.gpu_name = None

        if torch.cuda.is_available():

            try:

                self.gpu_name = (
                    torch.cuda.get_device_name(0)
                )

            except Exception:

                self.gpu_name = "CUDA GPU"

        self._prime_cpu_measurement()

        # ----------------------------------------------------
        # INITIAL METRIC COLLECTION
        # ----------------------------------------------------

        self.update_system_metrics(
            force=True
        )

    # --------------------------------------------------------
    # Prime CPU measurement
    # --------------------------------------------------------

    def _prime_cpu_measurement(self):

        if not PSUTIL_AVAILABLE:

            return

        try:

            psutil.cpu_percent(
                interval=None
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # Register camera
    # --------------------------------------------------------

    def register_camera(
        self,
        camera_id
    ):

        if camera_id not in self.camera_stats:

            self.camera_stats[camera_id] = {

                "frames": 0,

                "unavailable_frames": 0,

                "processing_errors": 0,

                "start_time": time.time(),

                "last_frame_time": None,

                "fps": 0.0,

                "last_processing_ms": 0.0,

                "total_processing_ms": 0.0,

                "processing_samples": 0

            }

    # --------------------------------------------------------
    # Register successful frame
    # --------------------------------------------------------

    def register_frame(
        self,
        camera_id,
        processing_ms=0.0
    ):

        self.register_camera(
            camera_id
        )

        now = time.time()

        self.total_frames += 1

        stats = self.camera_stats[
            camera_id
        ]

        stats["frames"] += 1

        stats["last_frame_time"] = now

        stats["last_processing_ms"] = (
            float(processing_ms)
        )

        stats["total_processing_ms"] += (
            float(processing_ms)
        )

        stats["processing_samples"] += 1

        # ----------------------------------------------------
        # Lifetime camera FPS
        # ----------------------------------------------------

        elapsed = (
            now - stats["start_time"]
        )

        if elapsed > 0:

            stats["fps"] = (
                stats["frames"] / elapsed
            )

    # --------------------------------------------------------
    # Register unavailable frame
    # --------------------------------------------------------

    def register_unavailable_frame(
        self,
        camera_id
    ):

        self.register_camera(
            camera_id
        )

        self.total_unavailable_frames += 1

        self.camera_stats[
            camera_id
        ]["unavailable_frames"] += 1

    # --------------------------------------------------------
    # Register AI processing error
    # --------------------------------------------------------

    def register_processing_error(
        self,
        camera_id
    ):

        self.register_camera(
            camera_id
        )

        self.total_processing_errors += 1

        self.camera_stats[
            camera_id
        ]["processing_errors"] += 1

    # --------------------------------------------------------
    # GPU METRICS
    # --------------------------------------------------------

    def _update_gpu_metrics(self):

        if not torch.cuda.is_available():

            self.gpu_utilization = None

            self.gpu_memory_used_mb = None

            self.gpu_memory_total_mb = None

            self.gpu_temperature = None

            return

        # ----------------------------------------------------
        # VRAM usage through PyTorch
        # ----------------------------------------------------

        try:

            free_bytes, total_bytes = (
                torch.cuda.mem_get_info(0)
            )

            used_bytes = (
                total_bytes - free_bytes
            )

            self.gpu_memory_used_mb = (
                used_bytes / (1024 * 1024)
            )

            self.gpu_memory_total_mb = (
                total_bytes / (1024 * 1024)
            )

        except Exception:

            try:

                self.gpu_memory_used_mb = (

                    torch.cuda.memory_allocated(0)
                    / (1024 * 1024)

                )

                self.gpu_memory_total_mb = (

                    torch.cuda
                    .get_device_properties(0)
                    .total_memory
                    / (1024 * 1024)

                )

            except Exception:

                self.gpu_memory_used_mb = None

                self.gpu_memory_total_mb = None

        # ----------------------------------------------------
        # NVIDIA utilization / temperature
        #
        # IMPORTANT:
        # nvidia-smi is called only every metrics_interval
        # seconds rather than every video frame.
        # ----------------------------------------------------

        try:

            result = subprocess.run(

                [
                    "nvidia-smi",

                    "--query-gpu="
                    "utilization.gpu,"
                    "temperature.gpu",

                    "--format="
                    "csv,noheader,nounits"
                ],

                capture_output=True,

                text=True,

                timeout=1.0
            )

            if result.returncode == 0:

                line = (
                    result.stdout.strip()
                )

                if line:

                    values = [

                        value.strip()

                        for value in (
                            line.split(",")
                        )

                    ]

                    if len(values) >= 1:

                        try:

                            self.gpu_utilization = (
                                float(values[0])
                            )

                        except Exception:

                            pass

                    if len(values) >= 2:

                        try:

                            self.gpu_temperature = (
                                float(values[1])
                            )

                        except Exception:

                            pass

        except Exception:

            # Preserve the previous valid reading instead
            # of unnecessarily clearing it.

            pass

    # --------------------------------------------------------
    # UPDATE SYSTEM METRICS
    # --------------------------------------------------------

    def update_system_metrics(
        self,
        force=False
    ):

        now = time.time()

        # ----------------------------------------------------
        # THROTTLE EXPENSIVE METRIC COLLECTION
        # ----------------------------------------------------

        if (

            not force

            and self.last_update_time > 0

            and (
                now - self.last_update_time
            ) < self.metrics_interval

        ):

            return False

        # ----------------------------------------------------
        # SYSTEM FPS
        # ----------------------------------------------------

        elapsed = (
            now - self.start_time
        )

        if elapsed > 0:

            self.system_fps = (

                self.total_frames
                / elapsed

            )

        # ----------------------------------------------------
        # CPU / RAM
        # ----------------------------------------------------

        if PSUTIL_AVAILABLE:

            try:

                self.cpu_percent = (
                    psutil.cpu_percent(
                        interval=None
                    )
                )

            except Exception:

                pass

            try:

                memory = (
                    psutil.virtual_memory()
                )

                self.memory_percent = (
                    memory.percent
                )

                self.memory_used_gb = (

                    memory.used
                    / (1024 ** 3)

                )

                self.memory_total_gb = (

                    memory.total
                    / (1024 ** 3)

                )

            except Exception:

                pass

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        self._update_gpu_metrics()

        self.last_update_time = now

        return True

    # --------------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------------

    def get_camera_state(
        self,
        camera_id
    ):

        stats = self.camera_stats.get(
            camera_id
        )

        if stats is None:

            return "UNKNOWN"

        last_frame = (
            stats.get(
                "last_frame_time"
            )
        )

        if last_frame is None:

            return "NO DATA"

        age = (
            time.time() - last_frame
        )

        if age > 10.0:

            return "STALLED"

        if age > 5.0:

            return "DEGRADED"

        return "HEALTHY"

    # --------------------------------------------------------
    # CAMERA METRICS
    # --------------------------------------------------------

    def get_camera_metrics(
        self,
        camera_id
    ):

        stats = self.camera_stats.get(
            camera_id
        )

        if stats is None:

            return {

                "fps": 0.0,

                "frames": 0,

                "unavailable_frames": 0,

                "processing_errors": 0,

                "last_processing_ms": 0.0,

                "average_processing_ms": 0.0,

                "state": "UNKNOWN"

            }

        average_processing_ms = 0.0

        if (
            stats["processing_samples"]
            > 0
        ):

            average_processing_ms = (

                stats["total_processing_ms"]
                / stats["processing_samples"]

            )

        return {

            "fps": stats["fps"],

            "frames": stats["frames"],

            "unavailable_frames": (
                stats["unavailable_frames"]
            ),

            "processing_errors": (
                stats["processing_errors"]
            ),

            "last_processing_ms": (
                stats["last_processing_ms"]
            ),

            "average_processing_ms": (
                average_processing_ms
            ),

            "state": self.get_camera_state(
                camera_id
            )

        }

    # --------------------------------------------------------
    # CONSOLE REPORT
    # --------------------------------------------------------

    def print_report(
        self,
        force=False
    ):

        now = time.time()

        if (

            not force

            and (
                now - self.last_report_time
            ) < self.report_interval

        ):

            return

        # ----------------------------------------------------
        # Refresh metrics only when necessary
        # ----------------------------------------------------

        self.update_system_metrics(
            force=force
        )

        self.last_report_time = now

        print(
            "\n" + "=" * 60
        )

        print(
            "[SYSTEM PERFORMANCE MONITOR]"
        )

        print(
            "=" * 60
        )

        print(
            f"Uptime: "
            f"{now - self.start_time:.1f}s"
        )

        print(
            f"System FPS: "
            f"{self.system_fps:.2f}"
        )

        print(
            f"Frames processed: "
            f"{self.total_frames}"
        )

        print(
            f"Unavailable frames: "
            f"{self.total_unavailable_frames}"
        )

        print(
            f"AI processing errors: "
            f"{self.total_processing_errors}"
        )

        if PSUTIL_AVAILABLE:

            print(
                f"CPU usage: "
                f"{self.cpu_percent:.1f}%"
            )

            print(
                f"RAM usage: "
                f"{self.memory_used_gb:.2f} / "
                f"{self.memory_total_gb:.2f} GB "
                f"({self.memory_percent:.1f}%)"
            )

        else:

            print(
                "CPU/RAM: psutil unavailable"
            )

        if self.gpu_name:

            print(
                f"GPU: "
                f"{self.gpu_name}"
            )

        if self.gpu_utilization is not None:

            print(
                f"GPU utilization: "
                f"{self.gpu_utilization:.1f}%"
            )

        else:

            print(
                "GPU utilization: unavailable"
            )

        if self.gpu_memory_used_mb is not None:

            if self.gpu_memory_total_mb:

                print(
                    f"GPU VRAM: "
                    f"{self.gpu_memory_used_mb:.0f} / "
                    f"{self.gpu_memory_total_mb:.0f} MB"
                )

            else:

                print(
                    f"GPU VRAM used: "
                    f"{self.gpu_memory_used_mb:.0f} MB"
                )

        if self.gpu_temperature is not None:

            print(
                f"GPU temperature: "
                f"{self.gpu_temperature:.0f} C"
            )

        print(
            "-" * 60
        )

        print(
            "CAMERA PERFORMANCE"
        )

        print(
            "-" * 60
        )

        for camera_id in self.camera_stats:

            metrics = (
                self.get_camera_metrics(
                    camera_id
                )
            )

            print(

                f"{camera_id}: "
                f"{metrics['state']} | "
                f"FPS {metrics['fps']:.2f} | "
                f"Frames {metrics['frames']} | "
                f"Unavailable "
                f"{metrics['unavailable_frames']} | "
                f"Errors "
                f"{metrics['processing_errors']} | "
                f"AI "
                f"{metrics['last_processing_ms']:.1f} ms"

            )

        print(
            "=" * 60
        )

    # --------------------------------------------------------
    # DATA FOR HUD
    #
    # This function is intentionally lightweight.
    # --------------------------------------------------------

    def get_metrics(
        self
    ):

        self.update_system_metrics(
            force=False
        )

        return {

            "system_fps": (
                self.system_fps
            ),

            "cpu_percent": (
                self.cpu_percent
            ),

            "memory_percent": (
                self.memory_percent
            ),

            "memory_used_gb": (
                self.memory_used_gb
            ),

            "memory_total_gb": (
                self.memory_total_gb
            ),

            "gpu_utilization": (
                self.gpu_utilization
            ),

            "gpu_memory_used_mb": (
                self.gpu_memory_used_mb
            ),

            "gpu_memory_total_mb": (
                self.gpu_memory_total_mb
            ),

            "gpu_temperature": (
                self.gpu_temperature
            ),

            "total_frames": (
                self.total_frames
            ),

            "total_unavailable_frames": (
                self.total_unavailable_frames
            ),

            "total_processing_errors": (
                self.total_processing_errors
            )

        }


# ============================================================
# LOAD CAMERA CONFIGURATION
# ============================================================

def load_camera_config():

    if not os.path.exists(
        CONFIG_FILE
    ):

        raise FileNotFoundError(
            f"Camera configuration not found: "
            f"{CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    cameras = config.get(
        "cameras",
        []
    )

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
    performance=None,
    system_metrics=None
):

    import cv2

    frame_height, frame_width = (
        frame.shape[:2]
    )

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
            max(
                20,
                perimeter_line[0][1] - 12
            )
        ),
        (0, 0, 255),
        0.65,
        2

    )

    # ========================================================
    # CAMERA HEALTH DATA
    # ========================================================

    health_status = (
        health_status or {}
    )

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
                max(
                    0,
                    frame_width - 390
                ),
                10
            ),

            (
                frame_width - 10,
                130
            ),

            (20, 20, 20),
            -1

        )

        draw_text(

            frame,
            "AI PERFORMANCE",
            (
                frame_width - 375,
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
                frame_width - 375,
                55
            ),
            (0, 255, 0),
            0.48,
            1

        )

        draw_text(

            frame,
            f"PROCESS: {last_processing_ms:.1f} ms",
            (
                frame_width - 210,
                55
            ),
            (255, 255, 255),
            0.40,
            1

        )

        draw_text(

            frame,
            f"AVG: {average_processing_ms:.1f} ms",
            (
                frame_width - 375,
                80
            ),
            (255, 255, 255),
            0.43,
            1

        )

        draw_text(

            frame,
            f"FRAMES: {frames_processed}",
            (
                frame_width - 210,
                80
            ),
            (255, 255, 255),
            0.43,
            1

        )

    # ========================================================
    # SYSTEM RESOURCE PANEL
    # ========================================================

    if system_metrics is not None:

        cpu_percent = system_metrics.get(
            "cpu_percent",
            0.0
        )

        memory_percent = system_metrics.get(
            "memory_percent",
            0.0
        )

        gpu_utilization = system_metrics.get(
            "gpu_utilization"
        )

        gpu_memory_used = system_metrics.get(
            "gpu_memory_used_mb"
        )

        gpu_memory_total = system_metrics.get(
            "gpu_memory_total_mb"
        )

        gpu_temperature = system_metrics.get(
            "gpu_temperature"
        )

        panel_x = max(
            0,
            frame_width - 390
        )

        panel_y = 140

        panel_width = 380

        panel_height = 90

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

        draw_text(

            frame,
            "SYSTEM RESOURCES",
            (
                panel_x + 10,
                panel_y + 20
            ),
            (255, 255, 255),
            0.45,
            2

        )

        draw_text(

            frame,
            f"CPU: {cpu_percent:.1f}%",
            (
                panel_x + 10,
                panel_y + 43
            ),
            (255, 255, 255),
            0.40,
            1

        )

        draw_text(

            frame,
            f"RAM: {memory_percent:.1f}%",
            (
                panel_x + 105,
                panel_y + 43
            ),
            (255, 255, 255),
            0.40,
            1

        )

        if gpu_utilization is not None:

            gpu_text = (
                f"GPU: "
                f"{gpu_utilization:.1f}%"
            )

        else:

            gpu_text = "GPU: N/A"

        draw_text(

            frame,
            gpu_text,
            (
                panel_x + 200,
                panel_y + 43
            ),
            (0, 255, 0),
            0.40,
            1

        )

        if (
            gpu_memory_used is not None
            and gpu_memory_total
        ):

            vram_text = (
                f"VRAM: "
                f"{gpu_memory_used:.0f}/"
                f"{gpu_memory_total:.0f} MB"
            )

        elif gpu_memory_used is not None:

            vram_text = (
                f"VRAM: "
                f"{gpu_memory_used:.0f} MB"
            )

        else:

            vram_text = "VRAM: N/A"

        draw_text(

            frame,
            vram_text,
            (
                panel_x + 10,
                panel_y + 67
            ),
            (255, 255, 255),
            0.39,
            1

        )

        if gpu_temperature is not None:

            temp_text = (
                f"GPU TEMP: "
                f"{gpu_temperature:.0f} C"
            )

        else:

            temp_text = "GPU TEMP: N/A"

        draw_text(

            frame,
            temp_text,
            (
                panel_x + 210,
                panel_y + 67
            ),
            (255, 255, 255),
            0.39,
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

        entity_id = det.get(
            "entity_id"
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
                max(
                    20,
                    y1 - 10
                )
            ),
            color,
            0.50,
            2

        )

        # ====================================================
        # GLOBAL ENTITY ID
        # ====================================================

        if (
            label == "person"
            and entity_id is not None
        ):

            draw_text(

                frame,
                f"ENTITY: {entity_id}",
                (
                    x1,
                    max(
                        38,
                        y1 - 28
                    )
                ),
                (0, 255, 255),
                0.46,
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

            for (
                reason_index,
                reason
            ) in enumerate(
                displayed_reasons
            ):

                reason_text = str(
                    reason
                )

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

        track_part = str(
            track_id
        )

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

    try:

        import msvcrt

    except ImportError:

        msvcrt = None

    # ========================================================
    # KEYBOARD COMMAND READER
    # ========================================================

    def read_command_key():

        """
        Read a non-blocking command from either
        the OpenCV window or the Windows console.
        """

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key != 255:

            return chr(
                key
            ).lower()

        if (
            msvcrt is not None
            and msvcrt.kbhit()
        ):

            try:

                key = (
                    msvcrt.getwch()
                )

                if key:

                    return key.lower()

            except Exception:

                pass

        return None

    # ========================================================
    # GPU SETUP
    # ========================================================

    prepare_gpu_environment()

    # ========================================================
    # SYSTEM MONITOR
    # ========================================================

    # --------------------------------------------------------
    # IMPORTANT:
    # Metrics are sampled every 2 seconds.
    #
    # Console report is generated every 10 seconds.
    #
    # This prevents nvidia-smi from being executed on
    # every processed video frame.
    # --------------------------------------------------------

    system_monitor = (
        SystemPerformanceMonitor(
            metrics_interval=2.0,
            report_interval=10.0
        )
    )

    if PSUTIL_AVAILABLE:

        print(
            "[OK] System resource monitoring: ACTIVE."
        )

    else:

        print(
            "[WARNING] psutil not installed."
        )

        print(
            "[WARNING] CPU/RAM monitoring disabled."
        )

        print(
            "[WARNING] Install with:"
        )

        print(
            "          python -m pip install psutil"
        )

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    print(
        "\n[*] Loading camera configuration..."
    )

    cameras = load_camera_config()

    print(
        f"[*] Cameras configured: "
        f"{len(cameras)}"
    )

    # ========================================================
    # SECURE STORAGE
    # ========================================================

    storage = SecureStorageBuffer(
        DATABASE_PATH
    )

    # ========================================================
    # EVENT MANAGER
    # ========================================================

    event_manager = EventManager(
        default_cooldown=5.0
    )

    # ========================================================
    # INTELLIGENT ALERT MANAGEMENT
    # ========================================================

    alert_manager = AlertManager(
        cooldown=10
    )

    # ========================================================
    # AUTOMATIC EVIDENCE MANAGEMENT
    # ========================================================

    evidence_manager = EvidenceManager(
        base_directory="evidence"
    )

    # ========================================================
    # CAMERA HEALTH AND TAMPER MONITORING
    # ========================================================

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

    incident_intelligence = IncidentIntelligence(
        incident_correlator
    )

    print(
        "[OK] Incident reconstruction intelligence: ACTIVE."
    )

    # ========================================================
    # CROSS-CAMERA ENTITY CORRELATION
    # ========================================================

    entity_correlator = CrossCameraCorrelator(
        similarity_threshold=0.80,
        max_entity_age=60.0
    )

    print(
        "[OK] Cross-camera entity correlation: ACTIVE."
    )

    print(
        "[OK] Global entity registry: ACTIVE."
    )

    # ========================================================
    # STREAM / ANALYTICS STORAGE
    # ========================================================

    streams = {}

    analytics_engines = {}

    # ========================================================
    # INITIALIZE CAMERAS
    # ========================================================

    for camera in cameras:

        if not camera.get(
            "enabled",
            True
        ):

            continue

        camera_id = camera["id"]

        source = camera["source"]

        system_monitor.register_camera(
            camera_id
        )

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
            # Separate analytics engine per camera
            # ------------------------------------------------

            analytics_engines[
                camera_id
            ] = BorderAnalyticsEngine(

                model_path=MODEL_PATH,

                use_gpu=True,

                perimeter_line=perimeter_line,

                loitering_seconds=(
                    loitering_seconds
                ),

                confidence_threshold=(
                    confidence_threshold
                ),

                iou_threshold=(
                    iou_threshold
                ),

                zones=camera.get(
                    "zones",
                    []
                )

            )

            print(
                f"[OK] {camera_id} online."
            )

        except Exception as exc:

            print(
                f"[ERROR] Could not initialize "
                f"{camera_id}: {exc}"
            )

            system_monitor.register_processing_error(
                camera_id
            )

    # ========================================================
    # VERIFY CAMERA AVAILABILITY
    # ========================================================

    if not streams:

        print(
            "[FATAL] No cameras could be started."
        )

        return

    # ========================================================
    # PLATFORM ONLINE
    # ========================================================

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
        "[*] Press R to show latest incident investigation report."
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
        "ACTIVE."
    )

    print(
        "[*] Global entity IDs: "
        "ACTIVE."
    )

    print(
        "[*] Intelligent incident correlation: "
        "ACTIVE."
    )

    print(
        "[*] System performance monitoring: "
        "ACTIVE."
    )

    print(
        "[*] System metrics sampling: "
        "every 2 seconds."
    )

    print(
        "[*] Performance report interval: "
        "every 10 seconds."
    )

    print(
        "[*] CPU / RAM / GPU monitoring: "
        "ACTIVE."
        if PSUTIL_AVAILABLE
        else
        "[*] CPU / RAM monitoring: "
        "LIMITED (install psutil)."
    )

    print("=" * 60)

    # ========================================================
    # MAIN LOOP
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
                # READ FRAME
                # ------------------------------------------------

                frame_start_time = (
                    time.perf_counter()
                )

                status, frame = (
                    stream.read()
                )

                # ------------------------------------------------
                # CAMERA HEALTH WHEN FRAME UNAVAILABLE
                # ------------------------------------------------

                if (
                    not status
                    or frame is None
                ):

                    system_monitor.register_unavailable_frame(
                        camera_id
                    )

                    health_status = (
                        health_monitor.update(
                            camera_id,
                            None
                        )
                    )

                    continue

                # ------------------------------------------------
                # INGESTION TIMESTAMP FOR STALE DETECTION
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

                    system_monitor.register_processing_error(
                        camera_id
                    )

                    print(
                        f"[AI ERROR] "
                        f"{camera_id}: {exc}"
                    )

                    continue

                # =================================================
                # CROSS-CAMERA ENTITY CORRELATION
                # =================================================

                try:

                    detections = (
                        entity_correlator.process_detections(

                            frame=frame,

                            camera_id=camera_id,

                            detections=detections

                        )
                    )

                except Exception as exc:

                    print(
                        f"[ENTITY ERROR] "
                        f"{camera_id}: {exc}"
                    )

                # =================================================
                # ATTACH GLOBAL ENTITY IDS TO EVENTS
                # =================================================

                detection_entity_map = {}

                for detection in detections:

                    if not isinstance(
                        detection,
                        dict
                    ):

                        continue

                    track_id = detection.get(
                        "track_id"
                    )

                    entity_id = detection.get(
                        "entity_id"
                    )

                    if (
                        track_id is not None
                        and entity_id is not None
                    ):

                        try:

                            detection_entity_map[
                                int(track_id)
                            ] = entity_id

                        except (
                            TypeError,
                            ValueError
                        ):

                            continue

                for event in events:

                    if not isinstance(
                        event,
                        dict
                    ):

                        continue

                    event_track_id = event.get(
                        "track_id"
                    )

                    if event_track_id is None:

                        continue

                    try:

                        entity_id = (
                            detection_entity_map.get(
                                int(event_track_id)
                            )
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        entity_id = None

                    if entity_id is not None:

                        event[
                            "entity_id"
                        ] = entity_id

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

                    entity_id = event.get(
                        "entity_id"
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

                    if risk_score is None:

                        risk_score = event.get(
                            "contextual_score"
                        )

                    risk_level = event.get(
                        "risk_level"
                    )

                    if risk_level is None:

                        risk_level = event.get(
                            "contextual_level"
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
                    # EventManager
                    # Prevent duplicate event generation.
                    # ------------------------------------------------

                    final_event = (
                        event_manager.create_event(

                            camera_id=camera_id,

                            event_type=event_type,

                            details=details,

                            track_id=track_id,

                            confidence=confidence,

                            risk_score=risk_score,

                            risk_level=risk_level,

                            zone=zone,

                            direction=direction

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

                        # IMPORTANT:
                        # This is the genuine global entity ID
                        # generated by EntityRegistry.
                        #
                        # It is NOT the local ByteTrack ID.

                        correlation_event[
                            "entity_id"
                        ] = entity_id

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
                            "risk_level"
                        ] = risk_level

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
                    # AUTOMATIC EVIDENCE CAPTURE
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

                                    # --------------------------------
                                    # Attach evidence to incident
                                    # --------------------------------

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
                    # STORE EVENT
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
                    # CONSOLE ALERT
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

                    if entity_id is not None:

                        print(
                            f"Global Entity ID: "
                            f"{entity_id}"
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
                # REGISTER FRAME PERFORMANCE
                # =================================================

                frame_processing_ms = (

                    time.perf_counter()
                    - frame_start_time

                ) * 1000.0

                system_monitor.register_frame(

                    camera_id,

                    frame_processing_ms

                )

                # =================================================
                # DISPLAY / HUD
                # =================================================

                display_frame = (
                    frame.copy()
                )

                # ------------------------------------------------
                # IMPORTANT:
                # get_metrics() uses cached values and only
                # refreshes expensive system/GPU metrics when
                # the 2-second sampling interval has elapsed.
                # ------------------------------------------------

                system_metrics = (
                    system_monitor.get_metrics()
                )

                draw_hud(

                    display_frame,

                    camera_id,

                    detections,

                    perimeter_line,

                    health_status,

                    analytics.get_performance(),

                    system_metrics

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
            # INCIDENT CLEANUP
            # ====================================================

            incident_correlator.cleanup()

            # ====================================================
            # ENTITY REGISTRY CLEANUP
            # ====================================================

            try:

                entity_correlator.cleanup()

            except Exception as exc:

                print(
                    f"[ENTITY CLEANUP ERROR] "
                    f"{exc}"
                )

            # ====================================================
            # SYSTEM PERFORMANCE UPDATE
            #
            # NOTE:
            # Deliberately NOT calling
            # update_system_metrics() here.
            #
            # get_metrics() above performs a throttled update.
            # This avoids unnecessary duplicate GPU queries.
            # ====================================================

            # ====================================================
            # PERIODIC SYSTEM REPORT
            # ====================================================

            system_monitor.print_report()

            # ====================================================
            # KEYBOARD COMMANDS
            # ====================================================

            # Q = quit
            # I = database integrity
            # E = recent events
            # S = incident statistics
            # R = latest investigation report

            command = read_command_key()

            # ----------------------------------------------------
            # Q = QUIT
            # ----------------------------------------------------

            if command == "q":

                print(
                    "\n[*] Q received. "
                    "Shutting down..."
                )

                break

            # ----------------------------------------------------
            # I = DATABASE INTEGRITY
            # ----------------------------------------------------

            elif command == "i":

                print(
                    "\n"
                    + "=" * 60
                )

                print(
                    "[SECURITY INTEGRITY]"
                )

                print(
                    "=" * 60
                )

                try:

                    valid, message = (
                        storage.verify_integrity()
                    )

                    print(

                        "[OK]" if valid
                        else "[CRITICAL]",

                        message

                    )

                except Exception as exc:

                    print(
                        "[INTEGRITY ERROR]",
                        exc
                    )

                print(
                    "=" * 60
                )

            # ----------------------------------------------------
            # E = RECENT EVENTS
            # ----------------------------------------------------

            elif command == "e":

                print(
                    "\n"
                    + "=" * 60
                )

                print(
                    "[RECENT EVENTS]"
                )

                print(
                    "=" * 60
                )

                try:

                    recent_events = (
                        storage.get_recent_events(
                            limit=10
                        )
                    )

                    if not recent_events:

                        print(
                            "No recent events found."
                        )

                    else:

                        for (
                            index,
                            stored_event
                        ) in enumerate(

                            recent_events,
                            1

                        ):

                            print(
                                f"{index}. "
                                f"{stored_event}"
                            )

                except Exception as exc:

                    print(
                        "[EVENT QUERY ERROR]",
                        exc
                    )

                print(
                    "=" * 60
                )

            # ----------------------------------------------------
            # S = INCIDENT STATISTICS
            # ----------------------------------------------------

            elif command == "s":

                print(
                    "\n"
                    + "=" * 60
                )

                print(
                    "[INCIDENT STATISTICS]"
                )

                print(
                    "=" * 60
                )

                try:

                    statistics = (
                        incident_correlator.get_statistics()
                    )

                    for name in (

                        "total_incidents",

                        "active_incidents",

                        "critical_incidents",

                        "high_incidents",

                        "medium_incidents"

                    ):

                        if name in statistics:

                            print(

                                f"{name.replace('_', ' ').title()}: "
                                f"{statistics[name]}"

                            )

                except Exception as exc:

                    print(
                        "[STATISTICS ERROR]",
                        exc
                    )

                print(
                    "=" * 60
                )

            # ----------------------------------------------------
            # R = LATEST INCIDENT REPORT
            # ----------------------------------------------------

            elif command == "r":

                print(
                    "\n"
                    + "=" * 60
                )

                print(
                    "[LATEST INCIDENT INVESTIGATION REPORT]"
                )

                print(
                    "=" * 60
                )

                try:

                    active_incidents = (
                        incident_intelligence
                        .get_active_incidents()
                    )

                    if not active_incidents:

                        print(
                            "No active incidents available."
                        )

                    else:

                        latest_incident = (
                            active_incidents[0]
                        )

                        latest_incident_id = (
                            latest_incident.get(
                                "incident_id"
                            )
                        )

                        if latest_incident_id:

                            incident_intelligence.print_investigation_report(

                                latest_incident_id

                            )

                        else:

                            print(
                                "Latest incident has no incident ID."
                            )

                except Exception as exc:

                    print(
                        "[REPORT ERROR]",
                        exc
                    )

                print(
                    "=" * 60
                )

            # ====================================================
            # CHECK IF ALL VIDEO FILES ENDED
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

        # ========================================================
        # FINAL PERFORMANCE REPORT
        # ========================================================

        try:

            system_monitor.print_report(
                force=True
            )

        except Exception as exc:

            print(
                "[PERFORMANCE REPORT ERROR]",
                exc
            )

        # ========================================================
        # SHUTDOWN
        # ========================================================

        print(
            "\n[*] Shutting down..."
        )

        for stream in streams.values():

            try:

                stream.stop()

            except Exception as exc:

                print(
                    f"[STREAM SHUTDOWN ERROR] "
                    f"{exc}"
                )

        cv2.destroyAllWindows()

        print(
            "[*] IBVAP shutdown complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()