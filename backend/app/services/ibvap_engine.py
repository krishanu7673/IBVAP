import threading
import time
from typing import Any, Dict, List, Optional


class IBVAPEngineService:
    """
    Service layer between the FastAPI backend and the existing
    IBVAP intelligence engine.

    The existing AI modules remain responsible for:
        - camera ingestion
        - object detection
        - tracking
        - analytics
        - event generation
        - incident correlation
        - evidence capture
        - storage

    This service will own the runtime lifecycle and expose
    application state to the web/API layer.
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._running = False
        self._started_at: Optional[float] = None

        self._worker_thread: Optional[threading.Thread] = None

        # Runtime components will be initialized in the next step.
        self._streams: Dict[str, Any] = {}
        self._analytics_engines: Dict[str, Any] = {}

        self._storage: Optional[Any] = None
        self._event_manager: Optional[Any] = None
        self._alert_manager: Optional[Any] = None
        self._evidence_manager: Optional[Any] = None
        self._health_monitor: Optional[Any] = None
        self._incident_correlator: Optional[Any] = None

        # API-visible state.
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._recent_events: List[Dict[str, Any]] = []
        self._active_incidents: Dict[str, Dict[str, Any]] = {}

        self._last_error: Optional[str] = None

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def start(self) -> Dict[str, Any]:
        """
        Start the IBVAP engine.

        Actual engine initialization and processing will be connected
        in the next integration step.
        """

        with self._lock:

            if self._running:
                return self.get_status()

            self._running = True
            self._started_at = time.time()
            self._last_error = None

        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """
        Stop the IBVAP engine.
        """

        with self._lock:
            self._running = False

        return self.get_status()

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current engine status.
        """

        with self._lock:

            uptime = 0.0

            if self._started_at is not None:
                uptime = time.time() - self._started_at

            return {
                "running": self._running,
                "uptime_seconds": round(uptime, 2),
                "camera_count": len(self._cameras),
                "active_incident_count": len(
                    self._active_incidents
                ),
                "recent_event_count": len(
                    self._recent_events
                ),
                "last_error": self._last_error,
            }

    # ============================================================
    # CAMERAS
    # ============================================================

    def get_cameras(self) -> List[Dict[str, Any]]:
        """
        Return API-visible camera information.
        """

        with self._lock:
            return list(self._cameras.values())

    def get_camera(
        self,
        camera_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Return one camera by ID.
        """

        with self._lock:
            return self._cameras.get(camera_id)

    # ============================================================
    # EVENTS
    # ============================================================

    def get_recent_events(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Return recent events maintained by the service.
        """

        with self._lock:

            return self._recent_events[
                -max(1, limit):
            ]

    # ============================================================
    # INCIDENTS
    # ============================================================

    def get_incidents(self) -> List[Dict[str, Any]]:
        """
        Return currently active incidents.
        """

        with self._lock:
            return list(
                self._active_incidents.values()
            )

    def get_incident(
        self,
        incident_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Return one active incident by ID.
        """

        with self._lock:
            return self._active_incidents.get(
                incident_id
            )

    # ============================================================
    # INTERNAL STATE HELPERS
    # ============================================================

    def _record_event(
        self,
        event: Dict[str, Any]
    ) -> None:
        """
        Add an event to the API-visible event buffer.
        """

        with self._lock:

            self._recent_events.append(
                dict(event)
            )

            # Keep the in-memory API buffer bounded.
            if len(self._recent_events) > 500:
                self._recent_events = (
                    self._recent_events[-500:]
                )

    def _record_incident(
        self,
        incident: Dict[str, Any]
    ) -> None:
        """
        Add/update an incident in the API-visible state.
        """

        incident_id = incident.get(
            "incident_id"
        )

        if not incident_id:
            return

        with self._lock:

            self._active_incidents[
                str(incident_id)
            ] = dict(incident)

    def _set_error(
        self,
        error: Exception | str
    ) -> None:
        """
        Store the latest engine error for API diagnostics.
        """

        with self._lock:
            self._last_error = str(error)