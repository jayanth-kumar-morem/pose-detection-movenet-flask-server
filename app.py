

from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app=Flask(__name__)
camera = cv2.VideoCapture(0)

model=tf.lite.Interpreter(model_path="./lite-model_movenet_singlepose_lightning_3.tflite")
model.allocate_tensors()

INPUT_SIZE=192
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_pts(kp,frame,conf):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(kp, [y,x,1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            img=frame.copy()
    
            # Dimension Change
            img=tf.expand_dims(img,axis=0)
            img=tf.cast(tf.image.resize_with_pad(img,INPUT_SIZE,INPUT_SIZE),dtype=tf.float32)

            # Model Prediction
            model.set_tensor(model.get_input_details()[0]['index'],img)
            model.invoke()
            kp=model.get_tensor(model.get_output_details()[0]['index'])

            # Draw Points
            draw_pts(kp,frame,0.4)

            # Draw Connection
            draw_connections(frame, kp, EDGES, 0.4)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(extra_files=["./templates"],debug=True,)