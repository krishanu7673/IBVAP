import cv2
import time


class SpatialIntelligence:

    def __init__(self, zones=None):
        self.zones = zones or []

        # Track the current zone of every tracked object.
        self.track_zones = {}

        # When an object entered its current zone.
        self.zone_entry_times = {}

        # Previous zone for transition detection.
        self.previous_zones = {}

        # Last time each track was observed.
        # This allows temporary tracking dropouts without
        # immediately destroying spatial history.
        self.track_last_seen = {}

    def _point_inside_polygon(self, point, polygon):

        if not polygon or len(polygon) < 3:
            return False

        try:
            polygon_points = [
                (int(x), int(y))
                for x, y in polygon
            ]

            result = cv2.pointPolygonTest(
                polygon_points,
                (float(point[0]), float(point[1])),
                False
            )

            return result >= 0

        except Exception:
            return False

    def _find_zone(self, center):
        """
        Find the most specific zone containing the point.

        If multiple polygons overlap, the later/more specific
        zone is allowed to take priority.
        """

        matched_zone = None

        for zone in self.zones:

            polygon = zone.get("polygon", [])

            if self._point_inside_polygon(
                center,
                polygon
            ):
                matched_zone = zone

        return matched_zone

    def analyze(
        self,
        track_id,
        center,
        current_time=None
    ):

        if current_time is None:
            current_time = time.time()

        zone = self._find_zone(center)

        if zone is None:

            current_zone_id = "OUTSIDE_DEFINED_ZONES"
            zone_name = "Outside Defined Zones"
            zone_type = "UNKNOWN"
            risk_weight = 0

        else:

            current_zone_id = zone.get(
                "id",
                "UNKNOWN_ZONE"
            )

            zone_name = zone.get(
                "name",
                current_zone_id
            )

            zone_type = zone.get(
                "type",
                "UNKNOWN"
            )

            try:
                risk_weight = float(
                    zone.get("risk_weight", 0)
                )

            except (
                TypeError,
                ValueError
            ):
                risk_weight = 0

        # ----------------------------------------------------------
        # Previous zone
        # ----------------------------------------------------------

        previous_zone_id = self.track_zones.get(
            track_id
        )

        # ----------------------------------------------------------
        # Zone transition detection
        # ----------------------------------------------------------

        zone_changed = (
            previous_zone_id is not None
            and previous_zone_id != current_zone_id
        )

        entered_zone = (
            zone_changed
            and current_zone_id != "OUTSIDE_DEFINED_ZONES"
        )

        exited_zone = (
            zone_changed
            and current_zone_id == "OUTSIDE_DEFINED_ZONES"
        )

        # ----------------------------------------------------------
        # Zone dwell timer
        # ----------------------------------------------------------

        if (
            track_id not in self.zone_entry_times
            or zone_changed
        ):
            self.zone_entry_times[track_id] = (
                current_time
            )

        dwell_time = (
            current_time
            - self.zone_entry_times[track_id]
        )

        # ----------------------------------------------------------
        # Update spatial history
        # ----------------------------------------------------------

        self.previous_zones[track_id] = (
            previous_zone_id
        )

        self.track_zones[track_id] = (
            current_zone_id
        )

        # ----------------------------------------------------------
        # Update last-seen time
        # ----------------------------------------------------------

        self.track_last_seen[track_id] = (
            current_time
        )

        return {
            "zone_id": current_zone_id,

            "zone_name": zone_name,

            "zone_type": zone_type,

            "risk_weight": risk_weight,

            "previous_zone_id": previous_zone_id,

            "zone_changed": zone_changed,

            "entered_zone": entered_zone,

            "exited_zone": exited_zone,

            "dwell_time": dwell_time
        }

    def get_current_zone(self, track_id):

        return self.track_zones.get(
            track_id
        )

    def get_zone(self, zone_id):

        for zone in self.zones:

            if zone.get("id") == zone_id:
                return zone

        return None

    def get_all_zones(self):

        return self.zones

    def cleanup(
        self,
        active_track_ids,
        max_age=30
    ):
        """
        Preserve spatial state during short tracking dropouts.

        A track can temporarily disappear from a detection frame.
        Its spatial history is therefore retained until it has been
        absent for more than max_age seconds.
        """

        active_track_ids = set(
            active_track_ids or []
        )

        current_time = time.time()

        # ----------------------------------------------------------
        # Update last-seen time for active tracks
        # ----------------------------------------------------------

        for track_id in active_track_ids:

            self.track_last_seen[track_id] = (
                current_time
            )

        # ----------------------------------------------------------
        # Find genuinely stale tracks
        # ----------------------------------------------------------

        stale_tracks = []

        for track_id in list(
            self.track_last_seen.keys()
        ):

            last_seen = self.track_last_seen.get(
                track_id,
                current_time
            )

            if (
                track_id not in active_track_ids
                and
                current_time - last_seen > max_age
            ):
                stale_tracks.append(
                    track_id
                )

        # ----------------------------------------------------------
        # Remove stale spatial state
        # ----------------------------------------------------------

        for track_id in stale_tracks:

            self.track_zones.pop(
                track_id,
                None
            )

            self.zone_entry_times.pop(
                track_id,
                None
            )

            self.previous_zones.pop(
                track_id,
                None
            )

            self.track_last_seen.pop(
                track_id,
                None
            )