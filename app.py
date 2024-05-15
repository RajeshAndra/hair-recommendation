from flask import Flask, render_template,Response,request,redirect,url_for
import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    detector = cv2.CascadeClassifier(r"Models\haarcascade_frontalface_default.xml")
    try:
        loaded_image=cv2.imread(r"static\Test Images\image_test.jpg")
        gray=cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY) 
        if loaded_image is not None:
            rect=detector.detectMultiScale(gray, 1.1, 5) 
            print(rect)
            if rect.size!=0:
                for (x,y,w,h) in rect: 
                    cropped_image=loaded_image[y:y+h+15, x:x+w]
                    cropped_image=cv2.resize(cropped_image,(300,300))
                    path = os.path.join("static\Test Images", "cropped_image_test.jpg")
                    cv2.imwrite(path,cropped_image)
                    return True  
        return False
    except Exception as e:
       return False
   
def shape():
    global face_shape
    plt.switch_backend('agg')
    interpreter = tf.lite.Interpreter(model_path=r"Models\face_shape_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_path="static\Test Images\cropped_image_test.jpg"
    image=cv2.imread(img_path)
    image=cv2.resize(image, (128, 128))
    x=np.array(image)/255
    input_data =  np.array(np.expand_dims(x, axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    
    output=np.array(output).flatten()
    l=[]
    labels=["Heart","Oblong","Oval","Round","Square"]
    for i in output:
        l.append(i/sum(output))
    plt.bar(labels,l,color ='blue',width=0.5)
    plt.savefig(r"C:\Users\rajur\Projects\Face Shape main\static\Test Images\graph.png")
    plt.close()
    face_shape=labels[np.argmax(l)]

def gender():
    global sex
    image=cv2.imread(r"static\Test Images\cropped_image_test.jpg")
    model=cv2.dnn.readNet(r"Models\gender_net.caffemodel",r"Models\gender_deploy.prototxt")
    genderList = ['Male', 'Female']
    
    blob = cv2.dnn.blobFromImage(image, 1, (227, 227), swapRB=False)
    model.setInput(blob)
    genderPreds = model.forward()
    sex = genderList[genderPreds[0].argmax()]
    print("Gender Output : {}".format(genderPreds))
    print("Gender : {}".format(sex))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
  
@app.route('/analyse')
def analyse():
    stat=crop_image()
    if not(stat):
        return render_template("index.html")
    shape()
    gender()
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
    elif request.form.get('Home') == 'Home':
        return redirect(url_for('index'))
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
    

camera.release()
cv2.destroyAllWindows()     