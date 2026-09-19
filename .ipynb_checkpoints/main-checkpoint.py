import cv2
import numpy as np

# Import custom modules from the src folder
from src.ingestion import VideoStream
from src.detection import ObjectDetector
from src.geofence import PolygonZoneDetector

def run_pipeline():
    # 1. Initialize Video Source (Video file path or RTSP link)
    video_source = "assets/test_border_feed.mp4" 
    stream = VideoStream(video_source)

    # 2. Define intrusion polygon coordinates (x, y points)
    polygon_points = np.array([
        [200, 200], 
        [800, 200], 
        [800, 600], 
        [200, 600]
    ])

    # 3. Initialize AI models and Geofence logic
    detector = ObjectDetector(model_path="yolov8n.pt")
    geofence = PolygonZoneDetector(polygon_pts=polygon_points, frame_resolution=(1280, 720))

    # 4. Processing Loop
    while True:
        ret, frame = stream.read_frame()
        if not ret:
            print("End of stream or video playback finished.")
            break

        # Detect and Track Objects
        tracked_objects = detector.process_frame(frame)

        # Check for perimeter intrusion
        intrusions = geofence.check_intrusion(tracked_objects)

        # Trigger visual alert on frame if an intrusion occurs
        if any(intrusions):
            cv2.putText(
                frame, 
                "WARNING: PERIMETER BREACH!", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                3
            )

        # Display output feed
        cv2.imshow("IBVAP Main Pipeline", frame)

        # Press 'q' to stop execution
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up stream
    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()