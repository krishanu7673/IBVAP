import os
import time
import subprocess

import psutil


class SystemMonitor:
    """
    System-wide IBVAP monitoring.

    Monitors:
    - CPU utilization
    - RAM utilization
    - GPU utilization
    - GPU VRAM
    - GPU temperature
    - processing statistics
    - camera statistics
    - entity statistics
    - incident statistics

    This module is monitoring-only.
    It does not modify detection, tracking,
    incident correlation, or evidence logic.
    """

    def __init__(
        self,
        report_interval=10.0
    ):
        self.report_interval = float(
            report_interval
        )

        self.start_time = time.time()

        self.last_report_time = (
            self.start_time
        )

        self.total_frames_processed = 0

        self.total_ai_processing_time = 0.0

        self.camera_stats = {}

        self.last_gpu_query = 0.0

        self.gpu_info = {
            "available": False,
            "name": "UNKNOWN",
            "utilization": 0.0,
            "memory_used_mb": 0.0,
            "memory_total_mb": 0.0,
            "memory_percent": 0.0,
            "temperature_c": 0.0
        }

        # Prime CPU measurement.
        try:
            psutil.cpu_percent(
                interval=None
            )
        except Exception:
            pass

    # ==========================================================
    # CAMERA REGISTRATION
    # ==========================================================

    def register_camera(
        self,
        camera_id
    ):
        camera_id = str(
            camera_id
        )

        if camera_id not in self.camera_stats:

            self.camera_stats[
                camera_id
            ] = {
                "frames": 0,
                "processing_time": 0.0,
                "errors": 0,
                "last_update": time.time()
            }

    # ==========================================================
    # FRAME UPDATE
    # ==========================================================

    def record_frame(
        self,
        camera_id,
        processing_time=0.0
    ):
        camera_id = str(
            camera_id
        )

        self.register_camera(
            camera_id
        )

        try:
            processing_time = float(
                processing_time
            )
        except (
            TypeError,
            ValueError
        ):
            processing_time = 0.0

        self.total_frames_processed += 1

        self.total_ai_processing_time += (
            processing_time
        )

        stats = self.camera_stats[
            camera_id
        ]

        stats["frames"] += 1

        stats["processing_time"] += (
            processing_time
        )

        stats["last_update"] = time.time()

    # ==========================================================
    # ERROR UPDATE
    # ==========================================================

    def record_error(
        self,
        camera_id
    ):
        camera_id = str(
            camera_id
        )

        self.register_camera(
            camera_id
        )

        self.camera_stats[
            camera_id
        ]["errors"] += 1

        self.camera_stats[
            camera_id
        ]["last_update"] = time.time()

    # ==========================================================
    # GPU MONITORING
    # ==========================================================

    def _query_gpu(self):

        now = time.time()

        # Avoid running nvidia-smi every frame.
        if (
            now - self.last_gpu_query
            < 2.0
        ):
            return self.gpu_info

        self.last_gpu_query = now

        command = [
            "nvidia-smi",
            "--query-gpu="
            "name,"
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "temperature.gpu",
            "--format=csv,noheader,nounits"
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                )
            )

            if result.returncode != 0:
                return self.gpu_info

            output = (
                result.stdout
                .strip()
            )

            if not output:
                return self.gpu_info

            line = output.splitlines()[0]

            parts = [
                part.strip()
                for part in line.split(",")
            ]

            if len(parts) < 5:
                return self.gpu_info

            name = parts[0]

            utilization = float(
                parts[1]
            )

            memory_used = float(
                parts[2]
            )

            memory_total = float(
                parts[3]
            )

            temperature = float(
                parts[4]
            )

            if memory_total > 0:

                memory_percent = (
                    memory_used
                    / memory_total
                    * 100.0
                )

            else:
                memory_percent = 0.0

            self.gpu_info = {
                "available": True,
                "name": name,
                "utilization": utilization,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "memory_percent": memory_percent,
                "temperature_c": temperature
            }

        except Exception:
            pass

        return self.gpu_info

    # ==========================================================
    # SNAPSHOT
    # ==========================================================

    def get_snapshot(
        self,
        active_cameras=0,
        active_tracks=0,
        active_entities=0,
        active_incidents=0
    ):

        now = time.time()

        uptime = (
            now
            - self.start_time
        )

        try:
            cpu_percent = psutil.cpu_percent(
                interval=None
            )
        except Exception:
            cpu_percent = 0.0

        try:

            memory = psutil.virtual_memory()

            ram_percent = (
                memory.percent
            )

            ram_used_gb = (
                memory.used
                / (
                    1024 ** 3
                )
            )

            ram_total_gb = (
                memory.total
                / (
                    1024 ** 3
                )
            )

        except Exception:

            ram_percent = 0.0
            ram_used_gb = 0.0
            ram_total_gb = 0.0

        gpu = self._query_gpu()

        if uptime > 0:

            system_fps = (
                self.total_frames_processed
                / uptime
            )

        else:
            system_fps = 0.0

        if (
            self.total_frames_processed
            > 0
        ):

            average_processing_ms = (
                self.total_ai_processing_time
                / self.total_frames_processed
                * 1000.0
            )

        else:
            average_processing_ms = 0.0

        return {
            "timestamp": now,

            "uptime_seconds": uptime,

            "cpu_percent": cpu_percent,

            "ram_percent": ram_percent,

            "ram_used_gb": ram_used_gb,

            "ram_total_gb": ram_total_gb,

            "gpu": dict(gpu),

            "total_frames_processed":
                self.total_frames_processed,

            "system_fps":
                system_fps,

            "average_processing_ms":
                average_processing_ms,

            "active_cameras":
                int(active_cameras),

            "active_tracks":
                int(active_tracks),

            "active_entities":
                int(active_entities),

            "active_incidents":
                int(active_incidents),

            "camera_stats":
                dict(self.camera_stats)
        }

    # ==========================================================
    # HEALTH STATUS
    # ==========================================================

    def get_health_status(
        self,
        snapshot
    ):

        cpu = snapshot.get(
            "cpu_percent",
            0.0
        )

        ram = snapshot.get(
            "ram_percent",
            0.0
        )

        gpu = snapshot.get(
            "gpu",
            {}
        )

        gpu_util = gpu.get(
            "utilization",
            0.0
        )

        gpu_temp = gpu.get(
            "temperature_c",
            0.0
        )

        status = "HEALTHY"

        warnings = []

        if cpu >= 90:

            status = "WARNING"

            warnings.append(
                "CPU utilization high"
            )

        if ram >= 90:

            status = "WARNING"

            warnings.append(
                "RAM utilization high"
            )

        if gpu_util >= 95:

            status = "WARNING"

            warnings.append(
                "GPU utilization high"
            )

        if gpu_temp >= 85:

            status = "WARNING"

            warnings.append(
                "GPU temperature high"
            )

        if ram >= 95:

            status = "CRITICAL"

            warnings.append(
                "RAM utilization critical"
            )

        if gpu_temp >= 95:

            status = "CRITICAL"

            warnings.append(
                "GPU temperature critical"
            )

        return {
            "status": status,
            "warnings": warnings
        }

    # ==========================================================
    # PERIODIC REPORT
    # ==========================================================

    def should_report(self):

        now = time.time()

        if (
            now - self.last_report_time
            >= self.report_interval
        ):

            self.last_report_time = now

            return True

        return False

    # ==========================================================
    # PRINT REPORT
    # ==========================================================

    def print_report(
        self,
        snapshot
    ):

        health = self.get_health_status(
            snapshot
        )

        gpu = snapshot.get(
            "gpu",
            {}
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "[SYSTEM MONITOR]"
        )

        print(
            f"Status: "
            f"{health['status']}"
        )

        print(
            f"Uptime: "
            f"{snapshot['uptime_seconds']:.1f}s"
        )

        print(
            f"CPU: "
            f"{snapshot['cpu_percent']:.1f}%"
        )

        print(
            f"RAM: "
            f"{snapshot['ram_used_gb']:.2f}/"
            f"{snapshot['ram_total_gb']:.2f} GB "
            f"({snapshot['ram_percent']:.1f}%)"
        )

        if gpu.get(
            "available",
            False
        ):

            print(
                f"GPU: "
                f"{gpu['name']}"
            )

            print(
                f"GPU Utilization: "
                f"{gpu['utilization']:.1f}%"
            )

            print(
                f"GPU VRAM: "
                f"{gpu['memory_used_mb']:.0f}/"
                f"{gpu['memory_total_mb']:.0f} MB "
                f"({gpu['memory_percent']:.1f}%)"
            )

            print(
                f"GPU Temperature: "
                f"{gpu['temperature_c']:.0f} C"
            )

        else:

            print(
                "GPU: UNAVAILABLE"
            )

        print(
            f"System FPS: "
            f"{snapshot['system_fps']:.2f}"
        )

        print(
            f"Average AI Processing: "
            f"{snapshot['average_processing_ms']:.2f} ms"
        )

        print(
            f"Active Cameras: "
            f"{snapshot['active_cameras']}"
        )

        print(
            f"Active Tracks: "
            f"{snapshot['active_tracks']}"
        )

        print(
            f"Active Entities: "
            f"{snapshot['active_entities']}"
        )

        print(
            f"Active Incidents: "
            f"{snapshot['active_incidents']}"
        )

        if health["warnings"]:

            print(
                "Warnings:"
            )

            for warning in health[
                "warnings"
            ]:

                print(
                    f"  - {warning}"
                )

        print(
            "=" * 60
        )