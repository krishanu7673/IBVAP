import sqlite3
import time
import hashlib


class SecureStorageBuffer:

    def __init__(
        self,
        db_path="ibvap_border_events.db"
    ):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(
            self.db_path
        )

    def _init_db(self):

        with self._connect() as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS border_events (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    timestamp REAL NOT NULL,

                    camera_id TEXT NOT NULL,

                    event_type TEXT NOT NULL,

                    track_id INTEGER,

                    confidence REAL,

                    details TEXT,

                    risk_score REAL,

                    zone TEXT,

                    direction TEXT,

                    snapshot_path TEXT,

                    incident_id TEXT,

                    prev_hash TEXT NOT NULL,

                    hash TEXT NOT NULL
                )
                """
            )

            # -------------------------------------------------
            # DATABASE MIGRATION
            # Adds new columns if an older database exists.
            # -------------------------------------------------

            cursor = conn.execute(
                "PRAGMA table_info(border_events)"
            )

            existing_columns = {
                row[1]
                for row in cursor.fetchall()
            }

            new_columns = {

                "risk_score":
                    "REAL",

                "zone":
                    "TEXT",

                "direction":
                    "TEXT",

                "snapshot_path":
                    "TEXT",

                "incident_id":
                    "TEXT"
            }

            for column_name, column_type in new_columns.items():

                if column_name not in existing_columns:

                    conn.execute(
                        f"""
                        ALTER TABLE border_events
                        ADD COLUMN {column_name} {column_type}
                        """
                    )

            conn.commit()

    def _get_last_hash(self, conn):

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT hash
            FROM border_events
            ORDER BY id DESC
            LIMIT 1
            """
        )

        row = cursor.fetchone()

        if row:
            return row[0]

        return "GENESIS_HASH"

    # ---------------------------------------------------------
    # HASH CALCULATION
    # ---------------------------------------------------------

    def _calculate_hash(
        self,
        timestamp,
        camera_id,
        event_type,
        track_id,
        confidence,
        details,
        risk_score,
        zone,
        direction,
        snapshot_path,
        incident_id,
        previous_hash
    ):

        # Normalize numeric values so that
        # 50 and 50.0 produce the same hash.

        if confidence is not None:
            confidence = float(confidence)

        if risk_score is not None:
            risk_score = float(risk_score)

        raw_data = (
            f"{timestamp}|"
            f"{camera_id}|"
            f"{event_type}|"
            f"{track_id}|"
            f"{confidence}|"
            f"{details}|"
            f"{risk_score}|"
            f"{zone}|"
            f"{direction}|"
            f"{snapshot_path}|"
            f"{incident_id}|"
            f"{previous_hash}"
        )

        return hashlib.sha256(
            raw_data.encode("utf-8")
        ).hexdigest()

    def log_event(
        self,
        camera_id,
        event_type,
        details,
        track_id=None,
        confidence=None,
        risk_score=None,
        zone=None,
        direction=None,
        snapshot_path=None,
        incident_id=None
    ):

        timestamp = time.time()

        with self._connect() as conn:

            previous_hash = self._get_last_hash(
                conn
            )

            # -------------------------------------------------
            # DATA USED FOR CRYPTOGRAPHIC HASH
            # -------------------------------------------------

            current_hash = self._calculate_hash(
                timestamp,
                camera_id,
                event_type,
                track_id,
                confidence,
                details,
                risk_score,
                zone,
                direction,
                snapshot_path,
                incident_id,
                previous_hash
            )

            # -------------------------------------------------
            # STORE EVENT
            # -------------------------------------------------

            conn.execute(
                """
                INSERT INTO border_events (

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id,

                    prev_hash,

                    hash

                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,

                (
                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id,

                    previous_hash,

                    current_hash
                )
            )

            conn.commit()

        return current_hash

    def verify_integrity(self):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id,

                    prev_hash,

                    hash

                FROM border_events

                ORDER BY id ASC
                """
            )

            rows = cursor.fetchall()

        previous_hash = "GENESIS_HASH"

        for row in rows:

            (
                event_id,

                timestamp,

                camera_id,

                event_type,

                track_id,

                confidence,

                details,

                risk_score,

                zone,

                direction,

                snapshot_path,

                incident_id,

                prev_hash,

                stored_hash

            ) = row

            # -------------------------------------------------
            # CHECK HASH CHAIN
            # -------------------------------------------------

            if prev_hash != previous_hash:

                return False, (
                    f"Broken hash chain at event "
                    f"{event_id}"
                )

            # -------------------------------------------------
            # RECREATE ORIGINAL HASH
            # -------------------------------------------------

            calculated_hash = self._calculate_hash(
                timestamp,
                camera_id,
                event_type,
                track_id,
                confidence,
                details,
                risk_score,
                zone,
                direction,
                snapshot_path,
                incident_id,
                prev_hash
            )

            # -------------------------------------------------
            # CHECK EVENT INTEGRITY
            # -------------------------------------------------

            if calculated_hash != stored_hash:

                return False, (
                    f"Tampering detected at "
                    f"event {event_id}"
                )

            previous_hash = stored_hash

        return True, (
            "Event chain verified successfully."
        )

    def get_recent_events(self, limit=20):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id

                FROM border_events

                ORDER BY id DESC

                LIMIT ?
                """,

                (limit,)
            )

            return cursor.fetchall()

    def get_event_by_id(self, event_id):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id,

                    prev_hash,

                    hash

                FROM border_events

                WHERE id = ?

                """,

                (event_id,)
            )

            return cursor.fetchone()

    def get_events_by_camera(
        self,
        camera_id,
        limit=50
    ):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id

                FROM border_events

                WHERE camera_id = ?

                ORDER BY id DESC

                LIMIT ?

                """,

                (
                    camera_id,
                    limit
                )
            )

            return cursor.fetchall()

    def get_events_by_incident(
        self,
        incident_id
    ):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id

                FROM border_events

                WHERE incident_id = ?

                ORDER BY timestamp ASC

                """,

                (incident_id,)
            )

            return cursor.fetchall()

    def get_high_risk_events(
        self,
        minimum_risk=70,
        limit=50
    ):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    id,

                    timestamp,

                    camera_id,

                    event_type,

                    track_id,

                    confidence,

                    details,

                    risk_score,

                    zone,

                    direction,

                    snapshot_path,

                    incident_id

                FROM border_events

                WHERE risk_score >= ?

                ORDER BY risk_score DESC, timestamp DESC

                LIMIT ?

                """,

                (
                    minimum_risk,
                    limit
                )
            )

            return cursor.fetchall()

    def count_events_by_type(self):

        with self._connect() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT

                    event_type,

                    COUNT(*)

                FROM border_events

                GROUP BY event_type

                ORDER BY COUNT(*) DESC

                """
            )

            return cursor.fetchall()