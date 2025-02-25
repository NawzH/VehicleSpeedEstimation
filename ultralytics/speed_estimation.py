from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import os
from datetime import datetime, timedelta

# ------------------ YOLO Model Setup --------------

# Loading the YOLO8m model.
model = YOLO("yolov8m.pt")

# Boolean value deciding if live camera feed will be used.
live_feed = False

if live_feed == True:
    # Chooses live camera feed.
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
else:
    # Chooses pre-recorded video.
    video_path = "C:/Users/nawza/Documents/GitHub/VehicleSpeedEstimation/ultralytics/data/IMG_6165.mov"
    cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"


# ------------- Frame Properties, FPS & Video Writer --------------

# FPS is handled differently for live vs pre recorded video.
if live_feed == True:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # Enables dynamic fps tracking.
    dynamic_fps = True
else:
    # Use fixed FPS from the video file if pre recorded video.
    dynamic_fps = False

# Get properties of the frame.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if live_feed == True:
    # If live camera feed, set fps to 30.
    fps = 30
else:
    fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer for pre recorded videos.
video_writer = None
if live_feed != True:
    video_writer = cv2.VideoWriter("VehicleSpeedEstimationVideo.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))


# ------------- Data Structures for Tracking & Colour Map --------------

# This stores the history of objects.
object_history = {}

# This stores the log of speeds.
speed_log = []

# Custom ID sequential mapping.
custom_id_counter = 1
custom_id_map = {}

# Defining the vertical line where speed is captured - X-Coordinate and minimum number of pixels.
speed_capture_x = 2700
speed_capture_threshold = 5
# Real world distance estimation which can be adjusted based on filming setup.
real_world_distance_m = 10
# Set the speed limit of the clip (in mph)
speed_limit = 30

# A predefined colour map for detecting colours of objects (black & white adjusted for shadows).
colour_map = {"black": (20,20,20),
              "white": (245, 245, 245),
              "red": (255, 0, 0),
              "blue": (0,0, 255),
              "green": (0, 255, 0),
              "yellow": (255, 255, 0),
              "orange": (255, 165, 0),
              "grey": (128, 128, 128),
              "silver": (192, 192, 192),
              "brown": (165, 42, 42)
            }


# --------- Loading Data from OBD Export (For Pre-Recorded) ----------
def load_obd_data(csv_file):
    data = []
    with open(csv_file, mode="r", encoding='utf8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Processes headers in the file
    header = [h.strip() for h in rows[1] if h.strip() != '']

    # Begin to iterate over actual OBD data rows
    for row in rows[2:]:
        if len(row) != len(header):
            continue

        row_data = dict(zip(header,row))
        local_time_raw = row_data.get('LocalTime', '').replace('="', '').replace('"', '')
        try:
            local_time = datetime.strptime(local_time_raw, "%Y-%m-%d %H:%M:%S.%f")
            speed = float(row_data.get('Speed[16]  (MPH)', '0').strip())
            data.append({'LocalTime': local_time, 'Speed': speed})
        except ValueError:
            continue

    print(f"loaded {len(data)} valid speeds from CSV") 
    return data

# Load OBD data
obd_data_path = "C:/Users/nawza/Documents/GitHub/VehicleSpeedEstimation/ultralytics/data/OBDdash_log_2025_2_25_10_37_5.csv"
obd_data = load_obd_data(obd_data_path)
if not obd_data:
    print("No valid OBD data found")
    exit()

# Get the video start time (using creation time)
video_start_time = datetime.fromtimestamp(os.path.getmtime(video_path))
print(f"Video start time: {video_start_time}")

obd_index = 0
while obd_index < len(obd_data) and obd_data[obd_index]['LocalTime'] < video_start_time:
    obd_index += 1

if obd_index == len(obd_data):
    print("No matching OBD data or incorrect creation time for the chosen video.")
    exit()

# ------------- Helper Functions --------------

# Function that finds the name of the closest colour.
def get_closest_colour(rgb):
    # The minimum distance is set to infinity at first to ensure real distance is smaller.
    min_distance = float("inf")
    # Default used if no colour match is found.
    closest_colour = "unknown"
    for colour_name, rgb_code in colour_map.items():
        # Uses the Euclidean distance between the pixel RGB and the colour currently iterating through.
        distance = np.linalg.norm(np.array(rgb) - np.array(rgb_code))
        # If the distance is smaller than the current minimum, the colour is updated.
        if distance < min_distance:
            min_distance = distance
            closest_colour = colour_name
    return closest_colour


# -------------- Main Processing Loop --------------

current_obd_speed = 0.0

# Process video
while cap.isOpened() and obd_index < len(obd_data):
    frame_start = time.time()
    success, frame = cap.read()
    if not success:
        print("The video frame is empty or video processing has finished")
        break

    # Handles dynamic fps for live video feed to ensure smooth video feed.
    if dynamic_fps == True:
        frame_end = time.time()
        frame_time = frame_end - frame_start
        if frame_time > 0:
            actual_fps = 1.0 / frame_time
        else:
            actual_fps = 30

    # This calculates the current frame timestamp
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_time = video_start_time + timedelta(seconds=frame_number/ fps)   

    # Updates the OBD speed if the new timestamp matches
    while obd_index < len(obd_data) and obd_data[obd_index]['LocalTime'] <= frame_time:
        current_obd_speed = obd_data[obd_index]['Speed']
        obd_index += 1

    # -------------- YOLO Object Detection & Tracking --------------

    # Run YOLO object detection and tracking.
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.65)

    # Speed capture line defined above is drawn on the video.
    cv2.line(frame, (speed_capture_x, 0), (speed_capture_x, frame_height), (0, 255, 255), 2)

    # Draw the OBD speed on the video at the top left corner.
    cv2.putText(frame, f"OBD Speed: {current_obd_speed:.2f} MPH", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Draw the bounding box around detected objects with tracking IDs.
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            # Get the coordinates of bounding box.
            x1, y1, x2, y2 = map(int, box)
            centre_x = (x1 + x2) / 2
            centre_y = (y1 + y2) / 2
            current_time = time.time()
            track_id = int(track_id)

            # Using YOLO label given to object, extract vehicle type.
            detected_class = model.names[int(class_id)]
            
            # Checks if the vehicle type is already assigned or type is unknown.
            if track_id not in object_history or object_history[track_id]['type'] == "unknown":
                # Then checks if YOLO detected a valid class, and if not, labels it as "uknown"
                if detected_class:
                    vehicle_type = detected_class
                else:
                    detected_class = "unknown"
            else:
                # Keeps existing vehicle type
                vehicle_type = object_history[track_id]['type']

            # Checks if the vehicle colour is already assigned or is unknown.
            if track_id not in object_history or object_history[track_id]['colour'] == "unknown":
                # Using the midpoint pixel of the bounding box, the vehicles colour is approximated.
                mid_x, mid_y = (x1+ x2) // 2, (y1 + y2) // 2
                if 0 <= mid_y < frame_height and 0 <= mid_x < frame_width:
                    # Gets the RGB value from the centre pixel of the bounding box.
                    pixel_rgb = frame[mid_y, mid_x]
                else:
                    # If midpoint is outside the frame, sets to default of black.
                    pixel_rgb = (0, 0,0)
                vehicle_colour = get_closest_colour(pixel_rgb)
            else:
                # Keeps existing colour of vehicle.
                vehicle_colour = object_history[track_id]['colour']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Custom ID is assigned to object. If not, placeholder is set until object crosses line.
            if track_id not in custom_id_map:
                custom_id_map[track_id] = None

            # If not in the object history, add to the dictionary.
            if track_id not in object_history:
                object_history[track_id] = {"positions": [], "speed": None, "crossed": False, "type": vehicle_type, "colour": vehicle_colour}

            # Store the objects movement history.
            object_history[track_id]["positions"].append((centre_x, centre_y, current_time))

            # Only the last two positions are kept for delta calculation.
            if len(object_history[track_id]["positions"]) > 2:
                object_history[track_id]["positions"].pop(0)

            # Now check if the object has crossed the speed capture line set.
            if not object_history[track_id]["crossed"]:
                prev_x, prev_y, prev_time = object_history[track_id]["positions"][0]
                curr_x, curr_y, curr_time = object_history[track_id]["positions"][-1]

                # Checks if the vehicle has crossed the speed capture line from right to the left.
                if prev_x > speed_capture_x >= curr_x:
                    # Calculates the pixel distance between the previous and current positions using Euclidean distance.
                    distance_in_px = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
                    # Calculates the time difference between the two positions
                    delta_time = curr_time - prev_time

                    if delta_time > 0 and distance_in_px > speed_capture_threshold:
                        # Converts the pixel movements to metres.
                        conversion_factor = real_world_distance_m / (x2 - x1)

                        # Calculates the object speed in mph.
                        object_speed_mph = (distance_in_px * conversion_factor) / delta_time * 2.24

                        # Custom ID is now assigned that its crossed the line/
                        if custom_id_map[track_id] is None:
                            custom_id_map[track_id] = custom_id_counter
                            custom_id_counter += 1
                        custom_id = custom_id_map[track_id]

                        # Speed of the object is stored.
                        object_history[track_id]["speed"] = object_speed_mph
                        object_history[track_id]["crossed"] = True

                        # Speed details of the vehicle are logged and added to the log file.
                        object_speed_entry = (
                            f"⏱ Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                            f"ID: {custom_id} | Speed: {object_speed_mph:.2f} mph | "
                            f"Type: {vehicle_type} | Colour: {vehicle_colour}"
                        )
                        speed_log.append(object_speed_entry)


            # Then display the unique ID, speed, type, and colour of the vehicle above bounding box.
            if object_history[track_id]["speed"] is not None:
                label = f"ID: {custom_id_map.get(track_id)}, Speed: {object_history[track_id]['speed']:.2f} mph, Type: {object_history[track_id]['type']}, Colour: {object_history[track_id]['colour']}"

                # Set colour of the bounding box based on the speed of the vehicle.
                if object_speed_mph <= speed_limit:
                    colour = (0, 255, 0)
                else:
                    colour = (0, 0, 255)

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # --------------- Show & save Frame --------------

    # Show the output.
    cv2.imshow('Vehicle Speed Estimation', frame)

    # If pre-recorded video is being used, video is saved.
    if not live_feed and video_writer is not None:
        video_writer.write(frame)
    
    # Allows user to press 'q' to exit the video feed from the model.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ------------ Cleanup and Save Logs --------------

# Speed log is saved to a text file
with open("speed_log.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(speed_log))

if video_writer == True:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()