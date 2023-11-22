import datetime
import os
import pickle
import shutil
import time
import uuid

import cv2
import face_recognition
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_cors import cross_origin
import pdfkit
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


ATTENDANCE_LOG_DIR = '.\\logs'
DB_PATH = '.\\db'

for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    file = request.files['file']
    file.filename = f"{uuid.uuid4()}.png"
    file.save(file.filename)

    user_name, match_status = recognize(cv2.imread(file.filename))

    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        # Ensure that the directories exist
        if not os.path.exists(ATTENDANCE_LOG_DIR):
            os.makedirs(ATTENDANCE_LOG_DIR)

        date_directory = os.path.join(ATTENDANCE_LOG_DIR, date)

        if not os.path.exists(date_directory):
            os.makedirs(date_directory)

        # Now, open the file for appending
        with open(os.path.join(date_directory, 'attendance.csv'), 'a') as f:
            f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'IN'))

        os.remove(file.filename)
    return jsonify({'user': user_name, 'match_status': match_status})


@app.route('/logout', methods=['POST'])
@cross_origin()
def logout():
    file = request.files['file']
    file.filename = f"{uuid.uuid4()}.png"
    file.save(file.filename)

    user_name, match_status = recognize(cv2.imread(file.filename))

    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        # Ensure that the directories exist
        if not os.path.exists(ATTENDANCE_LOG_DIR):
            os.makedirs(ATTENDANCE_LOG_DIR)

        date_directory = os.path.join(ATTENDANCE_LOG_DIR, date)

        if not os.path.exists(date_directory):
            os.makedirs(date_directory)

        # Now, open the file for appending
        with open(os.path.join(date_directory, 'attendance.csv'), 'a') as f:
            f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'OUT'))

        os.remove(file.filename)

    return jsonify({'user': user_name, 'match_status': match_status})


@app.route('/register_new_user', methods=['POST'])
@cross_origin()
def register_new_user():
    file = request.files['file']
    text = request.form.get('text')
    print(file)
    print(text)
    file.filename = f"{uuid.uuid4()}.png"
    file.save(file.filename)

    # Construct the full path for the pickle file
    pickle_file_path = os.path.join(DB_PATH, '{}.pickle'.format(text))

    # Check if the directory exists and create it if not
    os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

    embeddings = face_recognition.face_encodings(cv2.imread(file.filename))
    with open(pickle_file_path, 'wb') as file_:
        pickle.dump(embeddings, file_)

    os.remove(file.filename)

    return jsonify({'registration_status': 200})



@app.route('/get_attendance_logs', methods=['GET'])
@cross_origin()
def get_attendance_logs():
    filename = 'out.zip'
    shutil.make_archive(filename[:-4], 'zip', ATTENDANCE_LOG_DIR)
    return send_from_directory('.', filename, as_attachment=True)

@app.route('/', methods=['GET'])
@cross_origin()
def helloWorld():
    path_to_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_file("sample.html", "sample_file1.pdf",configuration=config)
    return "Hello World"

def recognize(img):
    embeddings_unknown = face_recognition.face_encodings(img)

    if len(embeddings_unknown) == 0:
        return 'no_persons_found', False

    embeddings_unknown = embeddings_unknown[0]

    best_match_score = 0
    best_match_name = 'unknown_person'

    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])

    for pickle_file in db_dir:
        path_ = os.path.join(DB_PATH, pickle_file)

        with open(path_, 'rb') as file:
            loaded_data = pickle.load(file)

        if isinstance(loaded_data, list) and len(loaded_data) > 0:
            embeddings = loaded_data[0]

            match_scores = face_recognition.face_distance([embeddings], embeddings_unknown)
            current_match_score = 1 - match_scores[0]  # Convert distance to a similarity score
            print(match_scores)
            if current_match_score > best_match_score:
                best_match_score = current_match_score
                best_match_name = pickle_file[:-7]  # Remove the '.pickle' extension
            print(best_match_name)
    # Decide based on the best match score
    if best_match_score >= 0.50:  # You can set a threshold for recognition accuracy
        return best_match_name, True
    else:
        return 'unknown_person', False

if __name__ == "__main__":
    app.run()