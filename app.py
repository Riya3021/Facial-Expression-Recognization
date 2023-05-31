from flask import Flask,render_template, request, redirect, url_for, session,Response
from camera import Video
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

from flask_sqlalchemy import SQLAlchemy
import cv2
from keras.models import load_model
import numpy as np
import sqlite3


from time import sleep
from tensorflow.keras.utils import load_img, img_to_array

face_classifier = cv2.CascadeClassifier(r'C:\Users\inspiron 5501\Desktop\G14 (project-1)\Facial expression recognition project\haarcascade_frontalface_alt2.xml')
classifier =load_model(r'C:\Users\inspiron 5501\Desktop\G14 (project-1)\Facial expression recognition project\model.h5')

emotion_labels = ['Angry','disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/coadings'
db = SQLAlchemy(app)



class contacts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    subject = db.Column(db.String(50), unique=False, nullable=False)
    message = db.Column(db.String(50), unique=False, nullable=False)

 
app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'coadings'
  
mysql = MySQL(app)

#website front page

@app.route('/')
def front():
    return render_template('pfrontendlog.html')

#webpage page

@app.route('/webpage')
def webpage():
    return render_template('pfrontend.html')

#emotion page

@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/expression')
def expression():
    return render_template('emotion1.html')


@app.route('/contactss')
def contactss():
    return render_template('contactss.html')


@app.route('/emotionss')
def emotionss():
    return render_template('pfrontendlog.html')

#home page

@app.route('/home')
def home():
    return render_template('pfrontend.html')


#about page

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/aboutus')
def aboutus():
    return render_template('about1.html')



#working page

@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/working')
def working():
    return render_template('working.html')


#contact page

@app.route('/contact', methods=['GET','POST'])
def contact(): 
    if(request.method=='POST'):

        name=request.form.get('name')
        email=request.form.get('email')
        sub=request.form.get('sub')
        msg=request.form.get('msg')



        entry=contacts(name=name,email=email, subject=sub , message=msg)

        db.session.add(entry)
        db.session.commit()
        
    return render_template('contact.html')


@app.route('/contactus', methods=['GET','POST'])
def contactus(): 
    if(request.method=='POST'):

        name=request.form.get('name')
        email=request.form.get('email')
        sub=request.form.get('sub')
        msg=request.form.get('msg')

        entry=contacts(name=name,email=email, subject=sub , message=msg)

        db.session.add(entry)
        db.session.commit()
        
    return render_template('contact1.html')

#login and logout page

@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''

    if request.method==False:
         return render_template('pfrontendlog.html')

    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()

        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('pfrontend.html', mesage = mesage)
    
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('front'))
  
#register page

@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''

    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()

        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'

        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'

        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
            return render_template('login.html') 
            
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)
    
#emotion recognition via image

@app.route('/index')
def image():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	image = request.files['select_file']

	image.save('Facial expression recognition project\static/file.jpg')

	image = cv2.imread('Facial expression recognition project\static/file.jpg')

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('Facial expression recognition project\haarcascade_frontalface_alt2.xml')

	faces = cascade.detectMultiScale(gray, 1.1, 3)

	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

		cropped = image[y:y+h, x:x+w]


	cv2.imwrite('Facial expression recognition project\static/after.jpg', image)
	try:
		cv2.imwrite('Facial expression recognition project\static/cropped.jpg', cropped)

	except:
		pass

	try:
		img = cv2.imread('Facial expression recognition project\static/cropped.jpg', 0)

	except:
		img = cv2.imread('Facial expression recognition project\static/file.jpg', 0)

	img = cv2.resize(img, (48,48))
	img = img/255

	img = img.reshape(1,48,48,1)

	model = load_model('Facial expression recognition project\model.h5')

	pred = model.predict(img)


	label_map = ['Angry','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']
	pred = np.argmax(pred)
	final_pred = label_map[pred]
   
   

	return render_template('predict.html', data=final_pred)



#emotion recognition via live

@app.route('/indexs')
def indexs():
    return render_template('indexs.html')


def gen(camera):
    while True:
    
        frame = camera.get_frame()
        yield (b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n'
    )

@app.route('/video')

def video():
    
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	app.run(debug=True)
