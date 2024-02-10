# import cv2
# model=cv2.dnn.readNet("Models\gender_net.caffemodel","Models\gender_deploy.prototxt")
# genderList = ['Male', 'Female']
# image=cv2.imread("static\Images\Side-Swept Bangs.jpg")
# blob = cv2.dnn.blobFromImage(image, 1, (227, 227), swapRB=False)
# model.setInput(blob)
# genderPreds = model.forward()
# gender = genderList[genderPreds[0].argmax()]
# print("Gender Output : {}".format(genderPreds))
# print("Gender : {}".format(gender))

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
# model = tf.keras.models.load_model('Models\Face_detection_CNN_8072.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("face_shape_model.tflite", "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=r"Models\face_shape_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
img_path="static\Test Images\cropped_image_test.jpg"
image=cv2.imread(img_path)
image=cv2.resize(image, (128, 128))
x=np.array(image)/255
input_data =  np.array(np.expand_dims(x, axis=0), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output = interpreter.get_tensor(output_details[0]['index'])
output=np.array(output).flatten()
print(output)

l=[]
labels=["Heart","Oblong","Oval","Round","Square"]
for i in output:
    l.append(i/sum(output))
plt.bar(labels,l,color ='blue',width=0.5)
#plt.savefig(r"C:\Users\rajur\Projects\Face Shape main\static\Test Images\graph.png")
plt.show()

face_shape=labels[l.index(max(l))]
print(face_shape)