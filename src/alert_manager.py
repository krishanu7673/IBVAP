import time


class AlertManager:

    def __init__(
        self,
        cooldown=10
    ):
        self.cooldown = cooldown
        self.last_alerts = {}

    # ---------------------------------------------------------
    # CHECK WHETHER ALERT CAN BE GENERATED
    # ---------------------------------------------------------

    def should_alert(
        self,
        camera_id,
        track_id,
        alert_type
    ):

        key = (
            camera_id,
            track_id,
            alert_type
        )

        current_time = time.time()

        last_time = self.last_alerts.get(
            key
        )

        if last_time is None:

            self.last_alerts[key] = (
                current_time
            )

            return True

        if (
            current_time -
            last_time
        ) >= self.cooldown:

            self.last_alerts[key] = (
                current_time
            )

            return True

        return False

    # ---------------------------------------------------------
    # REGISTER ALERT
    # ---------------------------------------------------------

    def register_alert(
        self,
        camera_id,
        track_id,
        alert_type
    ):

        key = (
            camera_id,
            track_id,
            alert_type
        )

        self.last_alerts[key] = time.time()

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def cleanup(
        self,
        max_age=300
    ):

        current_time = time.time()

        expired = []

        for key, timestamp in (
            self.last_alerts.items()
        ):

            if (
                current_time -
                timestamp
            ) > max_age:

                expired.append(key)

        for key in expired:

            del self.last_alerts[key]