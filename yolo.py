import numpy as numpy
import cv2
from ultralytics import YOLO
from flask import Flask,request,render_template,Response


model=YOLO(r"yolov5nu.pt")
faces=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")


app=Flask(__name__)
camera=cv2.VideoCapture(0)


def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        frame=cv2.flip(frame,1)
        results=model.track(frame,stream=False,persist=True)
        results=results[0].plot()
        ret,buffer=cv2.imencode('.jpg',results)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')  


@app.route('/predict',methods=["POST","GET"])
def video_feed():
      return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
  
@app.route('/stop_prediction',methods=["POST","GET"])
def video_feeds():
      return Response(camera.release())

if __name__=="__main__":
    app.run()


