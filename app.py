from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
import glob

app = Flask(__name__)

attendance_file = 'attendance.csv'
recorded_names = set()

if not os.path.exists(attendance_file) or os.stat(attendance_file).st_size == 0:
    with open(attendance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Time'])

known_images = []
known_names = []

# Load known images
for image_path in glob.glob("extracted_folder/**/*.jpg", recursive=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    for face_encoding in face_encodings:
        known_images.append(face_encoding)
        known_names.append(os.path.basename(os.path.dirname(image_path)))

video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # below, the code on right returns two things seen on the left
        success, frame = video_capture.read()
        if not success:
            break
        else:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                #below compare_faces returns a boolean value
                matches = face_recognition.compare_faces(known_images, face_encoding)
                name = "Unknown"
                #below line calculates the Euclidean distances between current face encoding with
                # all known face encodings
                face_distances = face_recognition.face_distance(known_images, face_encoding)
                best_match_index = np.argmin(face_distances)
                #below code simply calculates if True:, there r 4 True in our case with 4 samir images
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                if name not in recorded_names and name != "Unknown":
                    now = datetime.now()
                    current_date = now.strftime("%Y-%m-%d")
                    current_time = now.strftime("%H:%M:%S")
                    with open(attendance_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, current_date, current_time])
                    recorded_names.add(name)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            #below code will start after the for loop, once all the names for the frame are calculated one by one
            #below code is used for encoding
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    with open(attendance_file, 'r') as file:
        csv_reader = csv.reader(file)
        attendance_list = list(csv_reader)
    return jsonify(attendance_list)

if __name__ == "__main__":
    app.run(debug=True)
















