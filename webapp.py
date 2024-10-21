
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob


from ultralytics import YOLO


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/dashboard")
def dash():
    return render_template('dashboard.html')

@app.route("/dashboard", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)
                                               
            file_extension = f.filename.rsplit('.', 1)[1].lower() 
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                

                image = Image.open(io.BytesIO(frame))

                # Perform the detection
                yolo = YOLO('mix.pt')
                detections = yolo.predict(image, save=True, source='runs/detect')
                return display(f.filename)
            
            elif file_extension == 'mp4': 
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
                # initialize the YOLOv8 model here
                model = YOLO('yolo_v8.pt')
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break                                                      

                    # do YOLOv8 detection on the frame here
                    model = YOLO('mix.pt')
                    results = model(frame, save=True)  #working
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)
                    
                    # write the frame to the output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()            


            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"



# #The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ",directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

    else:
        return "Format file salah!"
        
        
        

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        






# function for accessing rtsp stream
@app.route("/rtsp_feed")
def rtsp_feed():
    cap = cv2.VideoCapture('rtsp://abcdef:123456@192.168.202.33:554/stream1')
    model = YOLO('mix.pt')

    def generate():
        frame_count = 0  # Initialize frame counter
        skip_frames = 20 

        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % skip_frames != 0:
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            img = Image.open(io.BytesIO(frame))
            
            results = model(img, save=True)
            print(results)

            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
            # Convert image to BGR for further processing
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
            
            # Encode frame to JPEG bytes
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')







# Function to start webcam and detect objects

@app.route("/webcam_feed")
def webcam_feed():
    #source = 0
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()
            print(type(frame))
            
            img = Image.open(io.BytesIO(frame))
 
            
            model = YOLO('mix.pt')
            results = model(img, save=True)              

            print(results)
            cv2.waitKey(1)

            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)


            if cv2.waitKey(1) == ord('q'):
                break

            # read image as BGR
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR) 
            
            # Encode BGR image to bytes so that cv2 will convert to RGB
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            #print(frame)
                

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    #model = torch.hub.load('.', 'custom','best.pt', source='local')
    model = YOLO('mix.pt')
    app.run(host="0.0.0.0", port=args.port,debug=True) 
