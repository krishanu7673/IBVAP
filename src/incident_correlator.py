import time
import uuid


class IncidentCorrelator:

    """
    Correlates multiple related security events into a single incident.

    The correlator does not replace the existing event manager or storage.

    It maintains an in-memory incident intelligence structure containing:

    - chronological incident timeline
    - event history
    - risk progression
    - peak-risk event
    - cameras involved
    - evidence references
    - incident duration
    - tracked-entity duration
    - incident status
    - entity / track identity
    """

    def __init__(
        self,
        correlation_window=60.0,
        incident_timeout=120.0
    ):
        self.correlation_window = float(
            correlation_window
        )

        self.incident_timeout = float(
            incident_timeout
        )

        self.incidents = {}

        self.active_incidents = {}

    # =========================================================
    # INCIDENT ID
    # =========================================================

    def _generate_incident_id(self):

        timestamp = time.strftime(
            "%Y%m%d-%H%M%S"
        )

        short_id = uuid.uuid4().hex[:6].upper()

        return f"INC-{timestamp}-{short_id}"

    # =========================================================
    # ENTITY / TRACK ID
    # =========================================================

    def _get_entity_id(self, event):

        """
        Resolve the identity associated with an event.

        Priority:

        1. Explicit entity_id
        2. Local track_id
        3. None

        This allows future cross-camera entity correlation while
        currently using the tracker ID when no global entity ID exists.
        """

        entity_id = event.get(
            "entity_id"
        )

        if entity_id is not None:
            return entity_id

        track_id = event.get(
            "track_id"
        )

        if track_id is not None:
            return track_id

        return None

    # =========================================================
    # TRACK LIFECYCLE
    # =========================================================

    def _get_track_first_seen(self, event):

        """
        Get the first observed timestamp of the tracked entity.

        This is supplied by BorderAnalyticsEngine's track lifecycle
        intelligence.

        Falls back to the event timestamp when lifecycle information
        is not available.
        """

        value = event.get(
            "first_seen"
        )

        if isinstance(
            value,
            (int, float)
        ):
            return float(value)

        lifecycle = event.get(
            "track_lifecycle"
        )

        if isinstance(
            lifecycle,
            dict
        ):

            value = lifecycle.get(
                "first_seen"
            )

            if isinstance(
                value,
                (int, float)
            ):
                return float(value)

        timestamp = event.get(
            "timestamp"
        )

        if isinstance(
            timestamp,
            (int, float)
        ):
            return float(timestamp)

        return time.time()

    def _get_track_last_seen(self, event):

        """
        Get the latest observed timestamp of the tracked entity.

        This is supplied by BorderAnalyticsEngine's track lifecycle
        intelligence.
        """

        value = event.get(
            "last_seen"
        )

        if isinstance(
            value,
            (int, float)
        ):
            return float(value)

        lifecycle = event.get(
            "track_lifecycle"
        )

        if isinstance(
            lifecycle,
            dict
        ):

            value = lifecycle.get(
                "last_seen"
            )

            if isinstance(
                value,
                (int, float)
            ):
                return float(value)

        timestamp = event.get(
            "timestamp"
        )

        if isinstance(
            timestamp,
            (int, float)
        ):
            return float(timestamp)

        return time.time()

    def _get_track_duration(self, event):

        """
        Get the tracked entity's observed lifetime.

        Preferred source:

        1. Explicit track_duration
        2. first_seen -> last_seen
        3. 0
        """

        duration = event.get(
            "track_duration"
        )

        if duration is not None:

            try:

                return max(
                    0.0,
                    float(duration)
                )

            except (
                TypeError,
                ValueError
            ):
                pass

        first_seen = (
            self._get_track_first_seen(
                event
            )
        )

        last_seen = (
            self._get_track_last_seen(
                event
            )
        )

        return max(
            0.0,
            last_seen - first_seen
        )

    # =========================================================
    # CORRELATION KEY
    # =========================================================

    def _get_correlation_key(self, event):

        """
        Prefer entity_id because it can connect observations
        across cameras.

        If no entity_id exists, fall back to camera + track.
        """

        entity_id = self._get_entity_id(
            event
        )

        if entity_id is not None:

            return (
                "ENTITY",
                str(entity_id)
            )

        camera_id = event.get(
            "camera_id"
        )

        track_id = event.get(
            "track_id"
        )

        if (
            camera_id is not None
            and
            track_id is not None
        ):

            return (
                "TRACK",
                str(camera_id),
                str(track_id)
            )

        if camera_id is not None:

            return (
                "CAMERA",
                str(camera_id)
            )

        return (
            "UNKNOWN",
        )

    # =========================================================
    # EVENT TIME
    # =========================================================

    def _get_event_time(self, event):

        timestamp = event.get(
            "timestamp"
        )

        if isinstance(
            timestamp,
            (int, float)
        ):

            return float(timestamp)

        return time.time()

    # =========================================================
    # EVENT TYPE
    # =========================================================

    def _get_event_type(self, event):

        event_type = event.get(
            "event_type"
        )

        if event_type is None:

            event_type = event.get(
                "type"
            )

        if event_type is None:

            event_type = "UNKNOWN_EVENT"

        return str(
            event_type
        )

    # =========================================================
    # RISK SCORE
    # =========================================================

    def _get_risk_score(
        self,
        event,
        fallback=0.0
    ):

        risk_score = event.get(
            "risk_score"
        )

        if risk_score is None:

            return float(
                fallback
            )

        try:

            return float(
                risk_score
            )

        except (
            TypeError,
            ValueError
        ):

            return float(
                fallback
            )

    # =========================================================
    # FIND ACTIVE INCIDENT
    # =========================================================

    def _find_matching_incident(
        self,
        correlation_key,
        event_time
    ):

        incident_id = self.active_incidents.get(
            correlation_key
        )

        if incident_id is None:

            return None

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            self.active_incidents.pop(
                correlation_key,
                None
            )

            return None

        last_event_time = incident.get(
            "last_event_time",
            incident.get(
                "created_at",
                event_time
            )
        )

        elapsed = (
            event_time
            - last_event_time
        )

        if elapsed > self.correlation_window:

            return None

        if elapsed > self.incident_timeout:

            return None

        return incident

    # =========================================================
    # CREATE INCIDENT
    # =========================================================

    def _create_incident(
        self,
        event,
        correlation_key,
        event_time
    ):

        incident_id = (
            self._generate_incident_id()
        )

        risk_score = (
            self._get_risk_score(
                event
            )
        )

        camera_id = event.get(
            "camera_id"
        )

        entity_id = (
            self._get_entity_id(
                event
            )
        )

        track_first_seen = (
            self._get_track_first_seen(
                event
            )
        )

        track_last_seen = (
            self._get_track_last_seen(
                event
            )
        )

        track_duration = (
            self._get_track_duration(
                event
            )
        )

        incident = {

            "incident_id":
                incident_id,

            "created_at":
                event_time,

            "updated_at":
                event_time,

            "last_event_time":
                event_time,

            "camera_id":
                camera_id,

            "entity_id":
                entity_id,

            "correlation_key":
                correlation_key,

            "status":
                "ACTIVE",

            "initial_risk":
                risk_score,

            "current_risk":
                risk_score,

            "peak_risk":
                risk_score,

            "event_count":
                0,

            "event_types":
                [],

            "cameras":
                [],

            "events":
                [],

            "timeline":
                [],

            "evidence":
                [],

            "risk_history":
                [],

            "peak_event":
                None,

            # -------------------------------------------------
            # TRACK LIFECYCLE
            # -------------------------------------------------

            "track_first_seen":
                track_first_seen,

            "track_last_seen":
                track_last_seen,

            "track_duration":
                track_duration,

            # -------------------------------------------------
            # INCIDENT CLOSURE
            # -------------------------------------------------

            "closed_at":
                None,

            "incident_duration":
                0.0
        }

        self.incidents[
            incident_id
        ] = incident

        self.active_incidents[
            correlation_key
        ] = incident_id

        return incident

    # =========================================================
    # UPDATE TRACK LIFECYCLE
    # =========================================================

    def _update_track_lifecycle(
        self,
        incident,
        event
    ):

        first_seen = (
            self._get_track_first_seen(
                event
            )
        )

        last_seen = (
            self._get_track_last_seen(
                event
            )
        )

        duration = (
            self._get_track_duration(
                event
            )
        )

        existing_first_seen = incident.get(
            "track_first_seen"
        )

        existing_last_seen = incident.get(
            "track_last_seen"
        )

        # -----------------------------------------------------
        # Preserve earliest first_seen
        # -----------------------------------------------------

        if isinstance(
            first_seen,
            (int, float)
        ):

            if (
                existing_first_seen is None
                or
                first_seen < existing_first_seen
            ):

                incident[
                    "track_first_seen"
                ] = first_seen

        # -----------------------------------------------------
        # Preserve latest last_seen
        # -----------------------------------------------------

        if isinstance(
            last_seen,
            (int, float)
        ):

            if (
                existing_last_seen is None
                or
                last_seen > existing_last_seen
            ):

                incident[
                    "track_last_seen"
                ] = last_seen

        # -----------------------------------------------------
        # Recalculate lifecycle duration
        # -----------------------------------------------------

        first_seen = incident.get(
            "track_first_seen"
        )

        last_seen = incident.get(
            "track_last_seen"
        )

        if (
            isinstance(
                first_seen,
                (int, float)
            )
            and
            isinstance(
                last_seen,
                (int, float)
            )
        ):

            incident[
                "track_duration"
            ] = max(
                0.0,
                float(last_seen)
                - float(first_seen)
            )

        else:

            incident[
                "track_duration"
            ] = duration

    # =========================================================
    # ADD EVENT
    # =========================================================

    def add_event(self, event):

        """
        Add an event to an existing incident or create
        a new incident.

        Returns the complete incident dictionary.
        """

        if event is None:

            return None

        if not isinstance(
            event,
            dict
        ):

            return None

        event_time = (
            self._get_event_time(
                event
            )
        )

        correlation_key = (
            self._get_correlation_key(
                event
            )
        )

        incident = (
            self._find_matching_incident(
                correlation_key,
                event_time
            )
        )

        if incident is None:

            incident = (
                self._create_incident(
                    event,
                    correlation_key,
                    event_time
                )
            )

        incident_id = incident[
            "incident_id"
        ]

        event_type = (
            self._get_event_type(
                event
            )
        )

        risk_score = (
            self._get_risk_score(
                event,
                incident.get(
                    "current_risk",
                    0.0
                )
            )
        )

        # =====================================================
        # RESOLVE ENTITY ID
        # =====================================================

        resolved_entity_id = (
            self._get_entity_id(
                event
            )
        )

        # =====================================================
        # UPDATE TRACK LIFECYCLE
        # =====================================================

        self._update_track_lifecycle(
            incident,
            event
        )

        # =====================================================
        # EVENT RECORD
        # =====================================================

        event_record = dict(
            event
        )

        event_record[
            "incident_id"
        ] = incident_id

        event_record[
            "correlated_at"
        ] = time.time()

        if (
            event_record.get(
                "entity_id"
            ) is None
            and
            resolved_entity_id is not None
        ):

            event_record[
                "entity_id"
            ] = resolved_entity_id

        # -----------------------------------------------------
        # Preserve lifecycle information on stored event
        # -----------------------------------------------------

        event_record[
            "track_first_seen"
        ] = incident.get(
            "track_first_seen"
        )

        event_record[
            "track_last_seen"
        ] = incident.get(
            "track_last_seen"
        )

        event_record[
            "track_duration"
        ] = incident.get(
            "track_duration",
            0.0
        )

        incident[
            "events"
        ].append(
            event_record
        )

        # =====================================================
        # TIMELINE RECORD
        # =====================================================

        timeline_record = {

            "timestamp":
                event_time,

            "event_type":
                event_type,

            "camera_id":
                event.get(
                    "camera_id"
                ),

            "track_id":
                event.get(
                    "track_id"
                ),

            "entity_id":
                resolved_entity_id,

            "confidence":
                event.get(
                    "confidence"
                ),

            "risk_score":
                risk_score,

            "risk_level":
                event.get(
                    "risk_level"
                ),

            "zone":
                event.get(
                    "zone"
                ),

            "direction":
                event.get(
                    "direction"
                ),

            "details":
                event.get(
                    "details"
                )
        }

        incident[
            "timeline"
        ].append(
            timeline_record
        )

        # =====================================================
        # EVENT COUNT
        # =====================================================

        incident[
            "event_count"
        ] += 1

        # =====================================================
        # EVENT TYPES
        # =====================================================

        if event_type not in incident[
            "event_types"
        ]:

            incident[
                "event_types"
            ].append(
                event_type
            )

        # =====================================================
        # CAMERAS INVOLVED
        # =====================================================

        camera_id = event.get(
            "camera_id"
        )

        if (
            camera_id is not None
            and
            camera_id not in incident[
                "cameras"
            ]
        ):

            incident[
                "cameras"
            ].append(
                camera_id
            )

        # =====================================================
        # RISK HISTORY
        # =====================================================

        incident[
            "risk_history"
        ].append(
            {
                "timestamp":
                    event_time,

                "risk_score":
                    risk_score,

                "event_type":
                    event_type
            }
        )

        # =====================================================
        # CURRENT RISK
        # =====================================================

        incident[
            "current_risk"
        ] = risk_score

        # =====================================================
        # PEAK RISK
        # =====================================================

        if risk_score > incident[
            "peak_risk"
        ]:

            incident[
                "peak_risk"
            ] = risk_score

            incident[
                "peak_event"
            ] = {

                "timestamp":
                    event_time,

                "event_type":
                    event_type,

                "risk_score":
                    risk_score,

                "risk_level":
                    event.get(
                        "risk_level"
                    ),

                "camera_id":
                    camera_id,

                "track_id":
                    event.get(
                        "track_id"
                    ),

                "entity_id":
                    resolved_entity_id,

                "details":
                    event.get(
                        "details"
                    )
            }

        # =====================================================
        # CAMERA / ENTITY FALLBACK
        # =====================================================

        if incident.get(
            "camera_id"
        ) is None:

            incident[
                "camera_id"
            ] = camera_id

        if incident.get(
            "entity_id"
        ) is None:

            incident[
                "entity_id"
            ] = resolved_entity_id

        # =====================================================
        # STATUS
        # =====================================================

        incident[
            "status"
        ] = self._calculate_status(

            risk_score=risk_score,

            event_type=event_type,

            peak_risk=incident.get(
                "peak_risk",
                0.0
            ),

            previous_status=incident.get(
                "status"
            )
        )

        # =====================================================
        # UPDATE TIMES
        # =====================================================

        incident[
            "updated_at"
        ] = time.time()

        incident[
            "last_event_time"
        ] = event_time

        # =====================================================
        # CURRENT INCIDENT DURATION
        # =====================================================

        incident[
            "incident_duration"
        ] = self._calculate_incident_duration(
            incident
        )

        self.active_incidents[
            correlation_key
        ] = incident_id

        return incident

    # =========================================================
    # INCIDENT STATUS
    # =========================================================

    def _calculate_status(
        self,
        risk_score,
        event_type=None,
        peak_risk=None,
        previous_status=None
    ):

        """
        Calculate incident severity.

        Critical event types always produce CRITICAL status.

        Peak risk is used so an incident cannot downgrade merely
        because a later event has a lower numerical risk score.
        """

        critical_event_types = {
            "PERIMETER_BREACH"
        }

        if event_type in critical_event_types:

            return "CRITICAL"

        try:

            risk_score = float(
                risk_score
            )

        except (
            TypeError,
            ValueError
        ):

            risk_score = 0.0

        try:

            peak_risk = float(
                peak_risk
            )

        except (
            TypeError,
            ValueError
        ):

            peak_risk = risk_score

        effective_risk = max(
            risk_score,
            peak_risk
        )

        if effective_risk >= 75:

            return "CRITICAL"

        if effective_risk >= 50:

            return "HIGH"

        if effective_risk >= 25:

            return "MEDIUM"

        if previous_status == "CRITICAL":

            return "CRITICAL"

        return "LOW"

    # =========================================================
    # INCIDENT DURATION
    # =========================================================

    def _calculate_incident_duration(
        self,
        incident
    ):

        """
        Calculate the duration of the incident event chain.

        This is deliberately separate from track duration.

        Incident duration:

            first incident event
            ->
            latest incident event / closure

        Track duration:

            first entity observation
            ->
            last entity observation
        """

        start_time = incident.get(
            "created_at"
        )

        end_time = incident.get(
            "closed_at"
        )

        if end_time is None:

            end_time = incident.get(
                "last_event_time",
                start_time
            )

        if start_time is None:

            return 0.0

        try:

            return max(
                0.0,
                float(end_time)
                - float(start_time)
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

    def get_incident_duration(
        self,
        incident_id
    ):

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            return 0.0

        return self._calculate_incident_duration(
            incident
        )

    # =========================================================
    # TRACK DURATION
    # =========================================================

    def get_track_duration(
        self,
        incident_id
    ):

        """
        Return the observed lifetime of the entity associated
        with the incident.
        """

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            return 0.0

        duration = incident.get(
            "track_duration",
            0.0
        )

        try:

            return max(
                0.0,
                float(duration)
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

    # =========================================================
    # INCIDENT SUMMARY
    # =========================================================

    def get_incident_summary(
        self,
        incident_id
    ):

        """
        Returns a dashboard-ready summary of an incident.
        """

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            return None

        incident_duration = (
            self.get_incident_duration(
                incident_id
            )
        )

        track_duration = (
            self.get_track_duration(
                incident_id
            )
        )

        return {

            "incident_id":
                incident.get(
                    "incident_id"
                ),

            "status":
                incident.get(
                    "status"
                ),

            "camera_id":
                incident.get(
                    "camera_id"
                ),

            "entity_id":
                incident.get(
                    "entity_id"
                ),

            "cameras":
                list(
                    incident.get(
                        "cameras",
                        []
                    )
                ),

            "event_count":
                incident.get(
                    "event_count",
                    0
                ),

            "event_types":
                list(
                    incident.get(
                        "event_types",
                        []
                    )
                ),

            "initial_risk":
                incident.get(
                    "initial_risk",
                    0.0
                ),

            "current_risk":
                incident.get(
                    "current_risk",
                    0.0
                ),

            "peak_risk":
                incident.get(
                    "peak_risk",
                    0.0
                ),

            "peak_event":
                incident.get(
                    "peak_event"
                ),

            # -------------------------------------------------
            # INCIDENT DURATION
            # -------------------------------------------------

            "duration_seconds":
                round(
                    incident_duration,
                    2
                ),

            "incident_duration_seconds":
                round(
                    incident_duration,
                    2
                ),

            # -------------------------------------------------
            # TRACK DURATION
            # -------------------------------------------------

            "track_duration_seconds":
                round(
                    track_duration,
                    2
                ),

            "track_first_seen":
                incident.get(
                    "track_first_seen"
                ),

            "track_last_seen":
                incident.get(
                    "track_last_seen"
                ),

            # -------------------------------------------------
            # EVIDENCE
            # -------------------------------------------------

            "evidence_count":
                len(
                    incident.get(
                        "evidence",
                        []
                    )
                ),

            "timeline":
                list(
                    incident.get(
                        "timeline",
                        []
                    )
                )
        }

    # =========================================================
    # ADD EVIDENCE
    # =========================================================

    def add_evidence(
        self,
        incident_id,
        evidence_path
    ):

        if not incident_id:

            return False

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            return False

        if not evidence_path:

            return False

        if evidence_path not in incident[
            "evidence"
        ]:

            incident[
                "evidence"
            ].append(
                evidence_path
            )

        incident[
            "updated_at"
        ] = time.time()

        return True

    # =========================================================
    # CLOSE INCIDENT
    # =========================================================

    def close_incident(
        self,
        incident_id
    ):

        incident = self.incidents.get(
            incident_id
        )

        if incident is None:

            return False

        # -----------------------------------------------------
        # Prevent repeatedly closing the same incident
        # -----------------------------------------------------

        if incident.get(
            "status"
        ) == "CLOSED":

            return True

        close_time = time.time()

        incident[
            "status"
        ] = "CLOSED"

        incident[
            "closed_at"
        ] = close_time

        incident[
            "updated_at"
        ] = close_time

        # -----------------------------------------------------
        # Final incident duration
        # -----------------------------------------------------

        incident[
            "incident_duration"
        ] = self._calculate_incident_duration(
            incident
        )

        # -----------------------------------------------------
        # Final track duration
        # -----------------------------------------------------

        track_first_seen = incident.get(
            "track_first_seen"
        )

        track_last_seen = incident.get(
            "track_last_seen"
        )

        if (
            isinstance(
                track_first_seen,
                (int, float)
            )
            and
            isinstance(
                track_last_seen,
                (int, float)
            )
        ):

            incident[
                "track_duration"
            ] = max(
                0.0,
                float(track_last_seen)
                - float(track_first_seen)
            )

        # -----------------------------------------------------
        # Remove from active incidents
        # -----------------------------------------------------

        correlation_key = incident.get(
            "correlation_key"
        )

        if (
            correlation_key is not None
            and
            self.active_incidents.get(
                correlation_key
            ) == incident_id
        ):

            self.active_incidents.pop(
                correlation_key,
                None
            )

        return True

    # =========================================================
    # GET INCIDENT
    # =========================================================

    def get_incident(
        self,
        incident_id
    ):

        return self.incidents.get(
            incident_id
        )

    # =========================================================
    # GET ACTIVE INCIDENTS
    # =========================================================

    def get_active_incidents(self):

        active = []

        for incident in self.incidents.values():

            if incident.get(
                "status"
            ) != "CLOSED":

                active.append(
                    incident
                )

        active.sort(
            key=lambda item:
                item.get(
                    "updated_at",
                    0
                ),
            reverse=True
        )

        return active

    # =========================================================
    # GET ALL INCIDENTS
    # =========================================================

    def get_all_incidents(self):

        incidents = list(
            self.incidents.values()
        )

        incidents.sort(
            key=lambda item:
                item.get(
                    "created_at",
                    0
                ),
            reverse=True
        )

        return incidents

    # =========================================================
    # CLEANUP
    # =========================================================

    def cleanup(self):

        """
        Automatically closes incidents that have not received
        a new correlated event for incident_timeout seconds.
        """

        current_time = time.time()

        for (
            correlation_key,
            incident_id
        ) in list(
            self.active_incidents.items()
        ):

            incident = self.incidents.get(
                incident_id
            )

            if incident is None:

                self.active_incidents.pop(
                    correlation_key,
                    None
                )

                continue

            last_event_time = incident.get(
                "last_event_time",
                incident.get(
                    "updated_at",
                    current_time
                )
            )

            if (
                current_time
                - last_event_time
            ) > self.incident_timeout:

                self.close_incident(
                    incident_id
                )

    # =========================================================
    # STATISTICS
    # =========================================================

    def get_statistics(self):

        total = len(
            self.incidents
        )

        active = 0
        critical = 0
        high = 0

        for incident in self.incidents.values():

            status = incident.get(
                "status"
            )

            if status != "CLOSED":

                active += 1

            if status == "CRITICAL":

                critical += 1

            elif status == "HIGH":

                high += 1

        return {

            "total_incidents":
                total,

            "active_incidents":
                active,

            "critical_incidents":
                critical,

            "high_incidents":
                high
        }