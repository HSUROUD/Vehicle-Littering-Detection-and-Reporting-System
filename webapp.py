"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
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
import sys
import pandas as pd

app = Flask(__name__)






@app.route("/")
def hello_world():
    return render_template('index.html')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)

def get_frame():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = predict_img.imgpath    
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

'''
#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"
'''
'''
@app.route('/frame_display')
def frame_display():
    folder_path = 'runs/detect'
    #subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    #latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    filename = predict_img.imgpath
    directory = folder_path+'/'+filename 
    print("printing directory: ",directory)  
    
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    key_frames = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # 檢查幀編號是否有效
    frame_number = 0
    if frame_number < 0 or frame_number >= len(key_frames):
        return "Invalid frame number"

    # 獲取當前幀的文件名
    current_frame = key_frames[frame_number]

    # 將當前幀發送到客戶端
    return send_from_directory(directory, current_frame)
'''
'''
@app.route('/frame_display')
def frame_display():
    #folder_path = 'runs/detect/5'
    folder_path = 'static'
    # directory = 'runs/detect/5'
    # directory = 'C:/Users/User/Desktop/yolov7/runs/detect/5'
    #directory = 'C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/runs/detect/5'
    filename_with_extension = os.path.basename(predict_img.imgpath)
    filename_without_extension, file_extension = os.path.splitext(filename_with_extension)
    directory = folder_path + '/' + filename_without_extension
    print("printing directory: ", directory)

    #file_extension = filename.rsplit('.', 1)[1].lower()
    #environ = request.environ
    key_frames = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # 获取要显示的图像数量，你可以根据需要修改这个逻辑
    image_count = len(key_frames)

    # 创建一个包含所有图像文件名的列表
    image_filenames = [f for f in key_frames]

    # print('debug--', file=sys.stderr)
    print("image_filenames: ", image_filenames)
    # 渲染模板并传递所需的数据
    return render_template('index.html', image_path=filename_without_extension, image_count=image_count, image_filenames=image_filenames)
'''


    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            shooting_time = request.form.get('shooting-time')
            shooting_location = request.form.get('shooting-location')
            print(shooting_time,shooting_location)
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                process = Popen(["python", "littering_detect.py", '--source', filepath, "--car_weights","weights/yolov7.pt", "--trash_weights","weights/trash.pt","--plate_weights","weights/plate.pt"], shell=True)
                process.wait()
                
                
            elif file_extension == 'mp4':
                process = Popen(["python", "littering_detect.py", '--source', filepath, "--car_weights","weights/yolov7.pt", "--trash_weights","weights/trash.pt","--plate_weights","weights/plate.pt"], shell=True)
                process.communicate()
                process.wait()

    folder_path = 'static'
    #subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    #latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    #image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    filename_with_extension = os.path.basename(f.filename)
    filename_without_extension, file_extension = os.path.splitext(filename_with_extension)
    #filename_without_extension = '5'
    directory = folder_path + '/' + filename_without_extension
    print("printing directory: ", directory)
    df = pd.read_csv(directory+'/'+filename_without_extension+'.csv')
    df['shooting_time'] = shooting_time
    df['shooting_location'] = shooting_location
    df.to_csv(directory+'/'+filename_without_extension+'.csv', index=False)
    #file_extension = filename.rsplit('.', 1)[1].lower()
    #environ = request.environ

    key_frames = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # 获取要显示的图像数量，你可以根据需要修改这个逻辑
    image_count = len(key_frames)

    # 创建一个包含所有图像文件名的列表
    image_filenames = [f for f in key_frames]

    # print('debug--', file=sys.stderr)
    print("image_filenames: ", image_filenames)
    # 渲染模板并传递所需的数据
    return render_template('index.html', image_path=filename_without_extension, image_count=image_count, image_filenames=image_filenames, csv_data=df.to_html())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    #model = torch.hub.load('.', 'custom','weights/yolov7.pt', source='local')
    #model.eval()
    app.run(host="0.0.0.0", port=args.port , debug=True)  # debug=True causes Restarting with stat

