from flask import Flask, render_template, request, jsonify
import json
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from scipy import stats
import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import arabic_reshaper
from bidi.algorithm import get_display
import requests
app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

#Global variables 
model = load_model('DynamicModel.h5')
actions = np.array(["أسمك ايه ؟","الحمد لله تمام","بتشتغل إيه ؟","بكام فلوس"
                        ,"تيجي معايا ؟","جيران" ,"صديق","عائلة","عريس","مشاء الله","مع السلامة","وحشتني"])

#Just a test script for testing the api in early phases


@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the video file from the request
    video_file = request.files['video']

    # Save the video file to disk
    video_path = 'video.mp4'
    video_file.save(video_path)

    # Extract the frames from the video
    frames = video_to_frames(video_path)
    print("Number of frames: ", len(frames))

    # Convert the frames to JPEG format
    #frames_jpeg = [cv2.imencode('.jpg', frame)[1].tobytes() for frame in frames]
    # Extract keypoints from frames
    sequence = []
    predictions = []
    sentence = []
    threshold = 0.7
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            resList = []
            for frame in frames:

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #print(results)
                
                # Draw landmarks
                #draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-60:]
                
                if len(sequence) == 60:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    reshaped_Text = arabic_reshaper.reshape(actions[np.argmax(res)])
                    arabic_Sentence = get_display(reshaped_Text)
                    resList.append(arabic_Sentence)
                    predictions.append(np.argmax(res))


                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                #Modify this if condition as the sentence list always has different format (bel3araby)
                                if actions[np.argmax(res)] != sentence[-1]:
                                    reshaped_Text = arabic_reshaper.reshape(actions[np.argmax(res)])
                                    arabic_Sentence = get_display(reshaped_Text)
                                    sentence.append(arabic_Sentence)
                                    #print(sentence[-1])
                            else:
                                    reshaped_Text = arabic_reshaper.reshape(actions[np.argmax(res)])
                                    arabic_Sentence = get_display(reshaped_Text)
                                    sentence.append(arabic_Sentence)
                                    #print(sentence[-1])


    #if some frames matches the specified threshold
    if(len(sentence) != 0):
        print(sentence[-1])
        print("Above threshold")
        reshaped_Text = arabic_reshaper.reshape(sentence[-1])

        #reshaped_Text = arabic_reshaper.reshape("عذرا, لم يتم التحقق جيدا من الجملة.")

        arabic_Sentence = get_display(reshaped_Text)
        return arabic_Sentence

    else:
        #if Nothing matches the threshold then choose the first 60 frames
        print(resList[0])
        print("First 60 frames")
        reshaped_Text = arabic_reshaper.reshape("عذرا, لم يتم التحقق جيدا من الجملة.")
        arabic_Sentence = get_display(reshaped_Text)
        return "عذرا, لم يتم التحقق جيدا من الجملة."
    
def video_to_frames(video_path, frame_interval=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to hold the frames
    frames = []

    # Initialize a variable to keep track of the frame count
    count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame was not successfully read, break out of the loop
        if not ret:
            break

        # Increment the frame count
        count += 1

        # If the current frame is a multiple of the frame interval, add it to the list of frames
        if count % frame_interval == 0:
            frames.append(frame)

    # Release the video capture object
    cap.release()

    # Return the list of frames
    return frames

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])



