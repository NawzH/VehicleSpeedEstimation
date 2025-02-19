from ultralytics import YOLO
import cv2

# Loading the YOLO8m model.
model = YOLO("yolov8m.pt")

# Boolean value deciding if live camera feed will be used.
live_feed = False

if live_feed == True:
    # Chooses live camera feed.
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
else:
    # Chooses pre-recorded video.
    cap = cv2.VideoCapture("C:/Users/nawza/Documents/GitHub/VehicleSpeedEstimation/ultralytics/data/30road.mp4")

assert cap.isOpened(), "Error reading video file"

# Get properties of the frame.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if live_feed == True:
    # If live camera feed, set fps to 30.
    fps = 30
else:
    fps = cap.get(cv2.CAP_PROP_FPS)

# Process video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("The video frame is empty or video processing has finished")
        break
    
    # Run YOLO object detection and tracking.
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw the bounding box around detected objects with tracking IDs.
    if results[0].boxes is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            # Get the coordinates of bounding box.
            x1, y1, x2, y2 = map(int, box)
            # Draw the bounding box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Then display the unique ID of the vehicle above bounding box.
            if track_id is not None:
                label = f"ID: {int(track_id)}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

    # Show the output.
    cv2.imshow('Vehicle Speed Estimation', frame)
    # Allows user to press 'q' to exit the video feed from the model.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()