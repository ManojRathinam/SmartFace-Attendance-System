import cv2
import os
from flask import Flask, redirect, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Defining Flask App
app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


# A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


# A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


# A function to evaluate precision, recall, F1 score, and accuracy
def evaluate_metrics():
    y_true, y_pred = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            face_array = resized_face.ravel()
            identified_person = identify_face(face_array.reshape(1, -1))[0]
            y_true.append(user)
            y_pred.append(identified_person)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy


# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()

    # Evaluate metrics
    precision, recall, f1, accuracy = evaluate_metrics()

    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


# List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    # if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)

    # Get the screen width and height
    # 3 corresponds to the width dimension in OpenCV
    screen_width = int(cap.get(3))
    # 4 corresponds to the height dimension in OpenCV
    screen_height = int(cap.get(4))

    ret = True

    # Set the video window size to full screen
    cv2.namedWindow('Attendance', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        'Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        return render_template('home.html', mess='Failed to open the webcam. Please make sure it is connected.')

    while 1:
        ret, frame = cap.read()

        # Check if the frame is valid
        if frame is None:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name,
                            frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break

        # Display the frame
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # print('Training Model')
    # train_model()

    # Evaluate accuracy after training
    # print('Evaluating Metrics')
    # precision, recall, f1, accuracy = evaluate_metrics()
    # print(f'Precision: {precision * 100:.2f}%')
    # print(f'Recall: {recall * 100:.2f}%')
    # print(f'F1 Score: {f1 * 100:.2f}%')
    # print(f'Accuracy: {accuracy * 100:.2f}%')

    # Display evaluation metrics on the home page
    # names, rolls, times, l = extract_attendance()
    # return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
    #                      datetoday2=datetoday2, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


# A route to display the pie chart
@app.route('/metrics_chart')
def metrics_chart():
    precision, recall, f1, accuracy = evaluate_metrics()

    # Create a pie chart
    labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    sizes = [precision, recall, f1, accuracy]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.1, 0, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels,
            colors=colors, autopct='%1.1f%%', startangle=140)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Evaluation Metrics')
    plt.savefig('static/metrics_chart.png')
    plt.close()

    return render_template('metrics_chart.html')


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
