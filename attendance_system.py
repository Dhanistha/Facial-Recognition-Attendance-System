import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime

# Initialize Video Capture
video_capture = cv2.VideoCapture(0)

# Load Known Faces
image_path = os.path.join("faces", "image.png")
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}.")
    exit()

shahrukh_image = face_recognition.load_image_file(image_path)
shahrukh_encoding = face_recognition.face_encodings(shahrukh_image)[0]

known_face_encodings = [shahrukh_encoding]
known_face_names = ["Shahrukh"]

# List of expected students (for attendance)
students = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file for attendance
csv_filename = f"{current_date}.csv"
f = open(csv_filename, "a+", newline="")
lnwriter = csv.writer(f)

# Write CSV header if empty
if os.stat(csv_filename).st_size == 0:
    lnwriter.writerow(["Name", "Time"])

print("Press 'q' to quit.")

while True: 
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Please check your camera.")
        break
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])
                print(f"{name} marked as present at {current_time}")

    # Display the frame
    cv2.imshow("Attendance System", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
f.close()
