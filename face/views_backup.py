from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import face_recognition
import re
import os
import numpy as np
from .models import Person

# Create your views here.
def index(request):
    return render(request, "face/index.html")

def getpic(request):
    cam = VideoCamera()
    # return HttpResponse(gen(cam))
    known_face_encodings = []
    known_face_names = []
    known_faces_filenames = []
    persons = Person.objects.values("Image")

    # Walk in the folder to add every file name to known_faces_filenames
    for (dirpath, dirnames, filenames) in os.walk('media/face/images/'):
        known_faces_filenames.extend(filenames)
        break

    # Walk in the folder
    for filename in known_faces_filenames:
        # Load each file
        face = face_recognition.load_image_file('media/face/images/' + filename)
        # Extract the name of each employee and add it to known_face_names
        known_face_names.append(re.sub("[0-9]",'', filename[:-4]))
        # Encode the face of every employee
        known_face_encodings.append(face_recognition.face_encodings(face)[0])
    
    # face_picture = face_recognition.load_image_file("media/face/images/Geralt.jpg")
    # Detect faces
    face_locations = face_recognition.face_locations(gen(cam))
    # Encode faces
    face_encodings = face_recognition.face_encodings(gen(cam), face_locations)
    
    # if not face_encodings:
    #     return HttpResponse("error")    

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding) 
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            # Give the detected face the name of the employee that match
            name = known_face_names[best_match_index]

    for person in persons:
        print(person)
    return render(request, "face/getpic.html", {"persons": persons})

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            break

def gen(camera):
    while True:
        frame = camera.get_frame()
        return frame