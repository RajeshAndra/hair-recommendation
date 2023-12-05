from flask import Flask, render_template,Response,request,redirect,url_for
import os
import cv2
import dlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import random

app=Flask(__name__)
camera=cv2.VideoCapture(0)

capture=0
face_shape=None
sex=None

def generate_frames():
    global capture
    while True:
        success,frame=camera.read()
        if  success:
            if(capture):
                capture=0
                p = os.path.join('static\Test Images', "image_test.jpg")
                cv2.imwrite(p, frame)
                
            try:
                _,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
            except Exception as e:
                pass

            
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def crop_image():
    detector = dlib.get_frontal_face_detector()
    try:
        loaded_image=cv2.imread("static\Test Images\image_test.jpg")
        if loaded_image is not None:
            rects=detector(loaded_image)
            if rects:
                for rect in rects:
                    cropped_image=loaded_image[rect.top()-40:rect.bottom()+20,rect.left()-10:rect.right()+10]
                    cropped_image=cv2.resize(cropped_image,(300,300))
                    path = os.path.join("static\Test Images", "cropped_image_test.jpg")
                    cv2.imwrite(path,cropped_image)
                    break
            else:
                crop_image()

    except Exception as e:
        crop_image()
   
def shape():
    global face_shape
    model=tf.keras.models.load_model("Models\Face_detection_CNN_8072.h5")
    img_path="static\Test Images\cropped_image_test.jpg"
    image=cv2.imread(img_path)
    image=cv2.resize(image, (128, 128))
    x=np.array(image)/255
    x = np.expand_dims(x, axis=0)
    plt.switch_backend('agg')
    output=model.predict(x)
    output=np.array(output).flatten()
    l=[]
    labels=["Heart","Oblong","Oval","Round","Square"]
    for i in output:
        l.append(i/sum(output))
    plt.bar(labels,l,color ='blue',width=0.5)
    plt.savefig(r"C:\Users\rajur\Projects\Face Shape main\static\Test Images\graph.png")
    plt.close()
    face_shape=labels[l.index(max(l))]

def gender():
    global sex
    model=tf.keras.models.load_model("Models\gender_model.h5")
    img_path="static\Test Images\cropped_image_test.jpg"
    image=cv2.imread(img_path,0)
    image=cv2.resize(image, (100, 100))
    x=np.array(image)/255
    x = np.expand_dims(x, axis=0)
    output=model.predict(x)
    output=np.array(output).flatten()
    if output[0]>output[1]:
        sex= "Male"
    else:
        sex= "Female"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
  
@app.route('/analyse')
def analyse():
    crop_image()
    t1=threading.Thread(target=shape)
    t2=threading.Thread(target=gender)
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print(face_shape,sex)

    data=pd.read_excel("Hairstyle dataset.xlsx")
    result=data[(data["Face Shape"]==face_shape) & (data["Gender"]==sex)]["Hairstyle"]
    result=list(result)
    random.shuffle(result)
    
    img_path="static\Images"
    r0=result[0].replace(" ","%20")
    r1=result[1].replace(" ","%20")
    r2=result[2].replace(" ","%20")
    img0=os.path.join(img_path,r0+".jpg")
    img1=os.path.join(img_path,r1+".jpg")
    img2=os.path.join(img_path,r2+".jpg")
    return render_template('final.html',img0=img0,img1=img1,img2=img2,n0=result[0],n1=result[1],n2=result[2],faceshape=face_shape)

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.form.get('Analyse') == 'Analyse':
        global capture
        capture=1
        return redirect(url_for('analyse'))
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
    

camera.release()
cv2.destroyAllWindows()     