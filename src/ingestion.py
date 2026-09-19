import cv2
import threading
import time


class ThreadedRTSPStream:
    """
    Threaded video/RTSP stream reader.

    Supports:
        - MP4/video files
        - webcam
        - RTSP/IP cameras

    The reader continuously keeps the newest frame available.
    """

    def __init__(self, source, camera_id="CAMERA_01"):

        self.source = source
        self.camera_id = camera_id

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(
                f"[{self.camera_id}] Unable to open video source: {source}"
            )

        self.lock = threading.Lock()
        self.status = False
        self.frame = None
        self.stopped = False

        # Read first frame immediately
        status, frame = self.cap.read()

        if not status or frame is None:
            self.cap.release()

            raise ValueError(
                f"[{self.camera_id}] Unable to read first frame from: {source}"
            )

        self.status = True
        self.frame = frame

        self.thread = None
        self.last_frame_time = time.time()

    def start(self):

        self.thread = threading.Thread(
            target=self.update,
            daemon=True,
            name=f"Stream-{self.camera_id}"
        )

        self.thread.start()

        return self

    def update(self):

        # Get video's original FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Use 30 FPS if FPS information is invalid
        if fps <= 0 or fps > 240:
            fps = 30.0

        frame_interval = 1.0 / fps

        while not self.stopped:

            if not self.cap.isOpened():
                self.status = False
                time.sleep(0.2)
                continue

            status, frame = self.cap.read()

            if status and frame is not None:

                with self.lock:
                    self.status = True
                    self.frame = frame
                    self.last_frame_time = time.time()

                # Maintain original video playback speed
                time.sleep(frame_interval)

            else:

                # For a video file, this normally means EOF.
                if isinstance(self.source, str):
                    self.status = False
                    self.stopped = True
                    break

                # For RTSP/webcam, temporarily wait and retry.
                self.status = False
                time.sleep(0.1)

    def read(self):

        with self.lock:

            if self.frame is None:
                return False, None

            return self.status, self.frame.copy()

    def is_healthy(self, timeout=5.0):
        """
        Returns True if a frame has been received recently.
        """

        return (
            self.status
            and (time.time() - self.last_frame_time) <= timeout
        )

    def stop(self):

        self.stopped = True

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap.isOpened():
            self.cap.release()