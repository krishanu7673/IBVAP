import time


class IncidentIntelligence:

    """
    Investigation and incident-reconstruction layer.

    This module does not modify the existing detection,
    analytics, event manager, storage, or incident correlator.

    It reads incident data and produces structured
    investigation-ready intelligence.
    """

    def __init__(self, incident_correlator):
        self.incident_correlator = incident_correlator

    # =========================================================
    # GET INCIDENT
    # =========================================================

    def get_incident(self, incident_id):

        return self.incident_correlator.get_incident(
            incident_id
        )

    # =========================================================
    # GET ACTIVE INCIDENTS
    # =========================================================

    def get_active_incidents(self):

        return self.incident_correlator.get_active_incidents()

    # =========================================================
    # GET ALL INCIDENTS
    # =========================================================

    def get_all_incidents(self):

        return self.incident_correlator.get_all_incidents()

    # =========================================================
    # INCIDENT SUMMARY
    # =========================================================

    def get_summary(self, incident_id):

        return self.incident_correlator.get_incident_summary(
            incident_id
        )

    # =========================================================
    # TIMELINE
    # =========================================================

    def get_timeline(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return []

        timeline = list(
            incident.get(
                "timeline",
                []
            )
        )

        timeline.sort(
            key=lambda item: item.get(
                "timestamp",
                0
            )
        )

        return timeline

    # =========================================================
    # RISK PROGRESSION
    # =========================================================

    def get_risk_progression(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return []

        risk_history = list(
            incident.get(
                "risk_history",
                []
            )
        )

        risk_history.sort(
            key=lambda item: item.get(
                "timestamp",
                0
            )
        )

        return risk_history

    # =========================================================
    # EVENT SEQUENCE
    # =========================================================

    def get_event_sequence(self, incident_id):

        timeline = self.get_timeline(
            incident_id
        )

        return [
            item.get(
                "event_type",
                "UNKNOWN_EVENT"
            )
            for item in timeline
        ]

    # =========================================================
    # CAMERAS INVOLVED
    # =========================================================

    def get_cameras(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return []

        return list(
            incident.get(
                "cameras",
                []
            )
        )

    # =========================================================
    # EVIDENCE
    # =========================================================

    def get_evidence(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return []

        return list(
            incident.get(
                "evidence",
                []
            )
        )

    # =========================================================
    # PEAK EVENT
    # =========================================================

    def get_peak_event(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return None

        return incident.get(
            "peak_event"
        )

    # =========================================================
    # INCIDENT DURATION
    # =========================================================

    def get_duration(self, incident_id):

        return self.incident_correlator.get_incident_duration(
            incident_id
        )

    # =========================================================
    # INCIDENT PROGRESSION
    # =========================================================

    def get_progression(self, incident_id):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return None

        timeline = self.get_timeline(
            incident_id
        )

        risk_history = self.get_risk_progression(
            incident_id
        )

        return {
            "incident_id": incident_id,
            "status": incident.get(
                "status"
            ),
            "initial_risk": incident.get(
                "initial_risk",
                0
            ),
            "current_risk": incident.get(
                "current_risk",
                0
            ),
            "peak_risk": incident.get(
                "peak_risk",
                0
            ),
            "event_count": len(
                timeline
            ),
            "event_sequence": [
                item.get(
                    "event_type"
                )
                for item in timeline
            ],
            "risk_progression": [
                item.get(
                    "risk_score",
                    0
                )
                for item in risk_history
            ]
        }

    # =========================================================
    # INVESTIGATION REPORT
    # =========================================================

    def build_investigation_report(
        self,
        incident_id
    ):

        incident = self.get_incident(
            incident_id
        )

        if incident is None:
            return None

        summary = self.get_summary(
            incident_id
        )

        timeline = self.get_timeline(
            incident_id
        )

        risk_progression = (
            self.get_risk_progression(
                incident_id
            )
        )

        evidence = self.get_evidence(
            incident_id
        )

        peak_event = self.get_peak_event(
            incident_id
        )

        duration = self.get_duration(
            incident_id
        )

        return {
            "incident_id": incident_id,

            "status": incident.get(
                "status"
            ),

            "entity_id": incident.get(
                "entity_id"
            ),

            "primary_camera": incident.get(
                "camera_id"
            ),

            "cameras_involved": self.get_cameras(
                incident_id
            ),

            "created_at": incident.get(
                "created_at"
            ),

            "updated_at": incident.get(
                "updated_at"
            ),

            "duration_seconds": round(
                duration,
                2
            ),

            "initial_risk": incident.get(
                "initial_risk",
                0
            ),

            "current_risk": incident.get(
                "current_risk",
                0
            ),

            "peak_risk": incident.get(
                "peak_risk",
                0
            ),

            "peak_event": peak_event,

            "event_count": len(
                timeline
            ),

            "event_types": list(
                incident.get(
                    "event_types",
                    []
                )
            ),

            "timeline": timeline,

            "risk_progression": risk_progression,

            "evidence": evidence,

            "evidence_count": len(
                evidence
            )
        }

    # =========================================================
    # FORMAT TIMESTAMP
    # =========================================================

    def format_timestamp(
        self,
        timestamp
    ):

        try:

            return time.strftime(
                "%H:%M:%S",
                time.localtime(
                    float(timestamp)
                )
            )

        except (
            TypeError,
            ValueError
        ):

            return "--:--:--"

    # =========================================================
    # PRINT TIMELINE
    # =========================================================

    def print_timeline(
        self,
        incident_id
    ):

        timeline = self.get_timeline(
            incident_id
        )

        if not timeline:

            print(
                "\n[INCIDENT TIMELINE] No events."
            )

            return

        print(
            "\n"
            + "=" * 70
        )

        print(
            "INCIDENT TIMELINE"
        )

        print(
            f"Incident ID: {incident_id}"
        )

        print(
            "=" * 70
        )

        for index, event in enumerate(
            timeline,
            start=1
        ):

            timestamp = self.format_timestamp(
                event.get(
                    "timestamp"
                )
            )

            event_type = event.get(
                "event_type",
                "UNKNOWN"
            )

            risk_score = event.get(
                "risk_score",
                0
            )

            risk_level = event.get(
                "risk_level",
                "UNKNOWN"
            )

            camera_id = event.get(
                "camera_id",
                "UNKNOWN"
            )

            zone = event.get(
                "zone",
                "UNKNOWN"
            )

            direction = event.get(
                "direction",
                "UNKNOWN"
            )

            print(
                f"\n{index:02d}. "
                f"{timestamp} | "
                f"{event_type}"
            )

            print(
                f"    Camera: {camera_id}"
            )

            print(
                f"    Risk: "
                f"{risk_score}/100 "
                f"{risk_level}"
            )

            print(
                f"    Zone: {zone}"
            )

            print(
                f"    Direction: {direction}"
            )

            details = event.get(
                "details"
            )

            if details:

                print(
                    f"    Details: {details}"
                )

        print(
            "=" * 70
        )

    # =========================================================
    # PRINT INVESTIGATION REPORT
    # =========================================================

    def print_investigation_report(
        self,
        incident_id
    ):

        report = self.build_investigation_report(
            incident_id
        )

        if report is None:

            print(
                f"[ERROR] Incident not found: "
                f"{incident_id}"
            )

            return

        print(
            "\n"
            + "=" * 70
        )

        print(
            "IBVAP INCIDENT INVESTIGATION REPORT"
        )

        print(
            "=" * 70
        )

        print(
            f"Incident ID: "
            f"{report['incident_id']}"
        )

        print(
            f"Status: "
            f"{report['status']}"
        )

        print(
            f"Entity ID: "
            f"{report['entity_id']}"
        )

        print(
            f"Primary Camera: "
            f"{report['primary_camera']}"
        )

        print(
            f"Cameras Involved: "
            f"{', '.join(map(str, report['cameras_involved']))}"
        )

        print(
            f"Duration: "
            f"{report['duration_seconds']:.2f}s"
        )

        print(
            f"Initial Risk: "
            f"{report['initial_risk']}/100"
        )

        print(
            f"Current Risk: "
            f"{report['current_risk']}/100"
        )

        print(
            f"Peak Risk: "
            f"{report['peak_risk']}/100"
        )

        print(
            f"Event Count: "
            f"{report['event_count']}"
        )

        print(
            f"Evidence Count: "
            f"{report['evidence_count']}"
        )

        print(
            "\nEvent Sequence:"
        )

        for index, event_type in enumerate(
            report["event_types"],
            start=1
        ):

            print(
                f"  {index}. {event_type}"
            )

        if report["peak_event"]:

            print(
                "\nPeak Risk Event:"
            )

            print(
                f"  "
                f"{report['peak_event'].get('event_type')}"
            )

            print(
                f"  Risk: "
                f"{report['peak_event'].get('risk_score')}/100"
            )

        print(
            "\nRisk Progression:"
        )

        for item in report[
            "risk_progression"
        ]:

            timestamp = self.format_timestamp(
                item.get(
                    "timestamp"
                )
            )

            print(
                f"  {timestamp} | "
                f"{item.get('event_type')} | "
                f"{item.get('risk_score')}/100"
            )

        print(
            "\nEvidence:"
        )

        if report["evidence"]:

            for evidence in report[
                "evidence"
            ]:

                print(
                    f"  - {evidence}"
                )

        else:

            print(
                "  None"
            )

        print(
            "=" * 70
        )

    # =========================================================
    # STATISTICS
    # =========================================================

    def get_statistics(self):

        return self.incident_correlator.get_statistics()