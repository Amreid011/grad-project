from flask import Flask, flash, render_template, request, redirect, url_for, session
import os
import cv2
import pandas as pd
import numpy as np
import face_recognition
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

EMPLOYEE_FILE = 'employees.xlsx'
ATTENDANCE_FILE = 'attendance.xlsx'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '1234'

def load_employee_data():
    return pd.read_excel(EMPLOYEE_FILE)

def load_attendance_data():
    return pd.read_excel(ATTENDANCE_FILE)


def init_employee_file():
    if not os.path.exists(EMPLOYEE_FILE):
        df = pd.DataFrame(columns=['ID', 'Name', 'Department', 'Position'])
        df.to_excel(EMPLOYEE_FILE, index=False)


def init_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time', 'Leave Time'])
        df.to_excel(ATTENDANCE_FILE, index=False)


def add_employee(emp_id, name, department, position):
    df = pd.read_excel(EMPLOYEE_FILE)
    if emp_id in df['ID'].values:
        return False  
    new_data = {'ID': emp_id, 'Name': name, 'Department': department, 'Position': position}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_excel(EMPLOYEE_FILE, index=False)
    return True


def capture_employee_photos(emp_id, name, num_photos=10):
    folder_name = f"dataset/{emp_id}_{name.replace(' ', '_')}"
    os.makedirs(folder_name, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("Failed to open camera!", "error")
        return
    count = 0
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capturing Photos - Press q to quit', frame)
        img_path = os.path.join(folder_name, f"{count+1}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def load_known_faces(dataset_path='dataset'):
    known_encodings, known_ids, known_names = [], [], []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        emp_id = folder.split('_')[0]
        emp_name = " ".join(folder.split('_')[1:])
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_encodings.append(encoding)
                known_ids.append(emp_id)
                known_names.append(emp_name)
    return known_encodings, known_ids, known_names


def register_attendance(emp_id, name):
    time_now = datetime.now().strftime('%H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time', 'Leave Time'])
    already_present = df[(df['ID'] == int(emp_id)) & (df['Date'] == today_date)]
    if already_present.empty:
        new_row = {'ID': emp_id, 'Name': name, 'Date': today_date, 'Attendance Time': time_now, 'Leave Time': ''}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)
        flash(f"Attendance recorded for {name} at {time_now}.", "success")
    else:
        flash(f"{name} has already recorded attendance today.", "info")


def register_leave(emp_id, name):
    time_now = datetime.now().strftime('%H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        idx = df[(df['ID'] == int(emp_id)) & (df['Date'] == today_date)].index
        if not idx.empty:
            df.at[idx[0], 'Leave Time'] = time_now
            df.to_excel(ATTENDANCE_FILE, index=False)
            flash(f"Leave recorded for {name} at {time_now}.", "success")
        else:
            flash(f"No attendance record found for {name} today.", "error")


def recognize_face_and_register(mode='attend'):
    known_encodings, known_ids, known_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        flash("Failed to open camera!", "error")
        return

    recognized = False
    unknown_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(face_distances) == 0:
                    continue

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    emp_id = int(known_ids[best_match_index])
                    name = known_names[best_match_index]
                    if mode == 'attend':
                        register_attendance(emp_id, name)
                    else:
                        register_leave(emp_id, name)
                    label_name = name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    recognized = True
                    break
                else:
                    unknown_count += 1

        cv2.imshow("Face Recognition - Press w to exit", frame)

        if recognized:
            cap.release()
            cv2.destroyAllWindows()
            return

        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not recognized:
        flash("Unknown face. No attendance recorded.", "error")



def calculate_attendance_metrics():
    employees = load_employee_data()
    attendance = load_attendance_data()

    employee_metrics = []

    for _, emp in employees.iterrows():
        emp_id = emp['ID']
        name = emp['Name']
        department = emp['Department']
        position = emp['Position']

        emp_attendance = attendance[attendance['ID'] == emp_id]
        days_attended = len(emp_attendance[emp_attendance['Attendance Time'].notnull()])
        days_absent = len(emp_attendance[emp_attendance['Attendance Time'].isnull()])

        total_late_seconds = 0
        total_overtime_seconds = 0

        for _, row in emp_attendance.iterrows():
            if pd.notnull(row['Attendance Time']):
                attendance_time = datetime.strptime(row['Attendance Time'], '%H:%M:%S')
                scheduled_time = datetime.strptime("09:00:00", '%H:%M:%S')
                if attendance_time > scheduled_time:
                    late_seconds = (attendance_time - scheduled_time).seconds
                    total_late_seconds += late_seconds
            if pd.notnull(row['Leave Time']):
                leave_time = datetime.strptime(row['Leave Time'], '%H:%M:%S')
                scheduled_end_time = datetime.strptime("17:00:00", '%H:%M:%S')
                if leave_time > scheduled_end_time:
                    overtime_seconds = (leave_time - scheduled_end_time).seconds
                    total_overtime_seconds += overtime_seconds

        late_hours = total_late_seconds // 3600
        late_minutes = (total_late_seconds % 3600) // 60

        overtime_hours = total_overtime_seconds // 3600
        overtime_minutes = (total_overtime_seconds % 3600) // 60

        if days_absent > 5:
            color_class = 'red'
        elif days_absent == 5:
            color_class = 'yellow'
        else:
            color_class = 'green'

        employee_metrics.append({
            'ID': emp_id,
            'Name': name,
            'Department': department,
            'Position': position,
            'Days Attended': days_attended,
            'Days Absent': days_absent,
            'Late Hours': f"{int(late_hours)} hours and {int(late_minutes)} minutes",
            'Overtime Hours': f"{int(overtime_hours)} hours and {int(overtime_minutes)} minutes",
            'Color Class': color_class
        })

    return employee_metrics







@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin'] = username
        return redirect(url_for('dashboard'))
    else:
        flash("Incorrect username or password!", "error")
        return redirect(url_for('home', error='Incorrect username or password!'))

@app.route('/dashboard')
def dashboard():
    
    if 'admin' in session:
        metrics = calculate_attendance_metrics()
        return render_template('dashboard.html' , metrics=metrics)
    else:
        
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

@app.route('/add', methods=['POST'])
def add():
    emp_id = int(request.form['id'])
    name = request.form['name']
    department = request.form['department']
    position = request.form['position']
    init_employee_file()
    if add_employee(emp_id, name, department, position):
        capture_employee_photos(emp_id, name)
        flash("Employee added and photos captured successfully!", "success")
    else:
        flash("Employee already exists!", "error")
    return redirect('/')

@app.route('/face_attend')
def face_attend():
    recognize_face_and_register(mode='attend')
    return redirect(url_for('home'))

@app.route('/face_leave')
def face_leave():
    recognize_face_and_register(mode='leave')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)



