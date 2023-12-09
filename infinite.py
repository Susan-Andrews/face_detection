# issue


import cv2
import argparse
import os
import threading
from queue import Queue
import pyglet
from pyglet.window import key
from moviepy.editor import VideoFileClip

def play_video(video_path, video_queue):
    video = VideoFileClip(video_path)
    video.preview(fps=24)
    video_queue.put("DONE")

def video_thread(video_path, video_queue):
    video = VideoFileClip(video_path)
    player = pyglet.media.Player()
    source = pyglet.media.StreamingSource()
    media_format = pyglet.media.get_format_video(video_path, ".mp4")
    video_stream = pyglet.media.StreamingSource()
    video_stream.attach(pyglet.media.load(video_path))
    player.queue(video_stream)
    player.play()

    def on_close(_):
        player.pause()
        video_queue.put("DONE")

    window = pyglet.window.Window(width=media_format.width, height=media_format.height)
    window.on_close = on_close

    @window.event
    def on_draw():
        window.clear()
        player.get_texture().blit(0, 0)

    pyglet.app.run()

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "assets/opencv_face_detector.pbtxt"
faceModel = "assets/opencv_face_detector_uint8.pb"
ageProto = "assets/age_deploy.prototxt"
ageModel = "assets/age_net.caffemodel"
genderProto = "assets/gender_deploy.prototxt"
genderModel = "assets/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
face_count = 0  # Initialize face count variable

# Assuming the video folder is in the same directory as the script
video_folder = os.path.join(os.path.dirname(__file__), "video")

# Create a queue for communication between threads
video_queue = Queue()

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
    else:
        face_count = len(faceBoxes)  # Update face count

        # Play a video from the folder if faces are detected
        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if video_files:
            video_to_play = os.path.join(video_folder, video_files[0])
            # Start a new thread for video playback
            threading.Thread(target=video_thread, args=(video_to_play, video_queue)).start()

            # Wait for video playback to complete
