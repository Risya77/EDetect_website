###UPLOAD LIBRARY###
import os
import cv2
import numpy as np
from flask_cors import CORS
from flask import Flask,request, render_template, jsonify,redirect,session,flash,url_for
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
from functools import wraps
from chat import get_response
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from time import sleep
import urllib.request
from werkzeug.utils import secure_filename
#ambil data dari deteksi vidio
import logging
import os
import sys
import json
 

app=Flask(__name__,template_folder="templates")

CORS(app)
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='edetect'
app.config['UPLOAD_FOLDER'] = 'D:/EmotionDetect/bgproject/application/static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
mysql=MySQL(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET'])
#***************************************************Admin*****************************************#
####### LOGIN #########
@app.route('/login',methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST':
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor(DictCursor)
        cur.execute("select * from users where email=%s and password=%s",(email,pwd))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["name"]
            flash('Login Successfully','success')
            return redirect('index')
        else:
            flash('Invalid Login. Try Again','danger')
    return render_template("login.html")
#***************************************************Admin*****************************************#

@app.route("/indexAdmin")
# @Login_dulu 
def indexAdmin():
    return render_template("indexAdmin.html")
  
#check if user logged in
def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
####### END LOGIN #########

@app.route("/index")
def index():
    return render_template("index.html")

####### LOGOUT #########
@app.route("/logout")
@is_logged_in
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))
####### END LOGOUT #########

###### DATA DOSEN ########
###TAMPIL DATA###
@app.route("/data",methods=['GET'])
@is_logged_in
def data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM dosen")
    row = cur.fetchall()
    cur.close()
    return render_template('data_dosen.html', dosen=row)
    # return render_template('indexAdmin.html')

###SAVE DATA###
@app.route('/simpan',methods=["POST"])
def simpan():
    nip = request.form['nip']
    nama = request.form['nama']
    email = request.form['email']
    smt = request.form['semester']
    mk = request.form['mk']
    password = request.form['password']   
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO dosen VALUES (null, %s, %s,%s, %s,%s, %s)", (str(nip), str(nama),str(email), str(smt),str(mk), str(password)))
    mysql.connection.commit()
    return redirect(url_for('data'))

###UPDATE DATA###
@app.route('/update', methods=["POST"])
def update():
    id_data = request.form['id']
    nip = request.form['nip']
    nama = request.form['nama']
    email = request.form['email']
    smt = request.form['semester']
    mk = request.form['mk']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("UPDATE dosen SET nip=%s,nama=%s,email=%s,semester=%s,mk=%s,password=%s WHERE id=%s", (nip,nama,email,smt,mk,password, id_data))
    mysql.connection.commit()
    return redirect(url_for('data'))

###HAPUS DATA###
@app.route('/hapus/<string:id_data>', methods=["GET"])
def hapus(id_data):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM dosen WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('data'))

###### DATA User ########
###TAMPIL DATA###
@app.route("/userdata",methods=['GET'])
@is_logged_in
def userdata():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM users")
    row = cur.fetchall()
    cur.close()
    return render_template('data_admin.html', admin=row)

###SAVE DATA###
@app.route('/saveuser',methods=["POST"])
def saveuser():
    nama = request.form['nama']
    email = request.form['email']
    password = request.form['password']   
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO users VALUES (null, %s, %s,%s)", (str(nama),str(email), str(password)))
    mysql.connection.commit()
    return redirect(url_for('userdata'))

###UPDATE DATA###
@app.route('/upuser', methods=["POST"])
def upuser():
    id_data = request.form['id']
    nama = request.form['nama']
    email = request.form['email']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET name=%s,email=%s,password=%s WHERE id=%s", (nama,email,password, id_data))
    mysql.connection.commit()
    return redirect(url_for('userdata'))

###HAPUS DATA###
@app.route('/deluser/<string:id_data>', methods=["GET"])
def deluser(id_data):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('userdata'))

###### DATA Hasil ########
###TAMPIL DATA###
@app.route("/hasildata",methods=['GET'])
@is_logged_in
def hasildata():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM ekspresi")
    row = cur.fetchall()
    cur.close()
    return render_template('data_hasil.html', hasil=row)

###HAPUS DATA###
@app.route('/delhasil/<string:id_data>', methods=["GET"])
def delhasil(id_data):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM ekspresi WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('hasildata'))

####### END DATA - DATA #########

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###### Deteksi Open Kamera ########
#***************** START DETEKSI VIDIO REALTIME*****************#
@app.route("/detection", methods=['GET','POST'])

def detection():

    face_classifier = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')
    classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')

    emotion_labels = ['angry','happy','neutral','sad']
    def setup_custom_logger():
            LOG_DIR = os.getcwd() + '/' + 'logs'
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            
            formatter = logging.Formatter(json.dumps({'time':'%(asctime)s', 'name': '%(name)s', 'level': '%(levelname)s', 'message': '%(message)s'}))
            handler = logging.FileHandler(LOG_DIR+'/log.json')
            handler.setFormatter(formatter)
            screen_handler = logging.StreamHandler(stream=sys.stdout)
            screen_handler.setFormatter(formatter)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.addHandler(screen_handler)
            return logger

    cap = cv2.VideoCapture(0)
    logger = setup_custom_logger()
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                logger.info(label)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template("index.html")
###### End Deteksi Open Kamera ########
   

###### Deteksi Get Picture ########
@app.route('/pict')
def pict():
    return render_template('pict.html')


@app.route("/pdetec", methods=['POST'])
@is_logged_in
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
        emotion_labels = ['angry','happy','neutral','sad']
        faceCascade = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')

        image = cv2.imread('D:/EmotionDetect/bgproject/application/static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        flash('Found {0} faces!'.format(len(faces)))  
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
            
                prediksi = cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imwrite('D:/EmotionDetect/bgproject/application/Hasil/faces.jpg', prediksi)
        
        gbr=cv2.imread ('D:/EmotionDetect/bgproject/application/Hasil/faces.jpg',0)    
        smt = request.form['semester']
        mk = request.form['mk']
        jml=len(faces)
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO ekspresi VALUES (null, %s,%s,%s,%s)", (int(smt),str(mk),int(jml),str(label)))
        mysql.connection.commit()   
        return render_template('pict.html', filename=filename)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/hasil/<filename>')
def pdetect(filename):

    classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
    emotion_labels = ['angry','happy','neutral','sad']
    faceCascade = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')

    image = cv2.imread('D:/EmotionDetect/bgproject/application/static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)      
        else:
            cv2.putText(image,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
    
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)

###### End Deteksi Get Picture ######## 


    

###### Chatboot ######## 
@app.route("/chatbot", methods=['GET','POST'])
def chatbot():
    return render_template ("base.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    text =  request.get_json().get('message')
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)
###### End Chatboot ######## 

@app.route("/about", methods=['GET'])
def about():
    return render_template ("about.html")



if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(host='0.0.0.0', port=5000, debug=True)
