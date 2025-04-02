# Vehicle Speed Estimation Using Onboard Video and OBD-II Data
This project is a proof-of-concept for an On-board Vehicle Traffic Enforcement (OVTE) system. It uses computer vision to detect vehicles in a video feed, estimates their relative speed, and compares it with the host vehicle's speed (from OBD-II logs) to flag speeding infractions.

---

## Key Features

- Detects and tracks vehicles using YOLOv8.
- Supports pre-recorded or live video feeds.
- Integrates host vehicle speed from OBD-II data.
- Calculates and displays:
  - Relative speed of surrounding vehicles
  - Estimated actual speed (host + relative)
  - Vehicle type, colour, and unique ID
- Logs speeding infractions with time and details.

---

## Requirements

- Python 3.8 or later
- Dependencies listed below (install via pip)
- OBD-II Bluetooth scanner and [OBD Dash Pro](https://www.obddash.pro/) for exporting speed logs

## Installation
- Python 3.8 or later
- pip install -r requirements.txt

## How to Run
Option 1: Pre-recorded Video with OBD Data
- Open 'speed_estimation.py' and set 'live_feed' to false
- Place your video and exported OBD '.csv' file in ultralytics/data/
- Update the file paths in the script to match your filenames
- Run the program with python 'speed_estimation.py'

Option 2: Live Camera Feed (No OBD Integration)
- Open 'speed_estimation.py' and set 'live_feed' to true
- Ensure your webcam is connected (you may need to update the 'camera_index')
- Run the program with python 'speed_estimation.py'
- Note: Real-time OBD data streaming is not yet supported. Speed comparison is available only with pre-recorded video and exported OBD data.
