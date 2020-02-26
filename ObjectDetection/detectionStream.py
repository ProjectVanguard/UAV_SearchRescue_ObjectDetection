# Reference pyimagesearch.com

# USAGE
# python uavJoin.py --ip 0.0.0.0 --port 8000 --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
fps = FPS().start()
frameID = 0
global fpsStarted , oldFPS
fpsStarted = datetime.datetime.now()
oldFPS = None

# initialize a flask object
app = Flask(__name__)

# global variables for objects
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def objectDetection(frame_count):

    print("-------------------------Entered Object Detection--------------")
    global vs ,  outputFrame ,lock , fpsStarted , oldFPS , frameID
    # model
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    while True:
        # with lock:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)


        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        timestamp = datetime.datetime.now()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        elapsedTIME = timestamp - fpsStarted
        fpsFormat= "FPS: {}"
        if elapsedTIME.seconds >= 1:
            fps.stop()
            cv2.putText(frame,fpsFormat.format( fps.fps() ),
                (10 , 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                2)
            oldFPS = fps.fps()
            fpsStarted = fps.start()._start
            fps._numFrames = 0
        else:
            cv2.putText(frame,fpsFormat.format(oldFPS),
                (10 , 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                2)
        cv2.putText(frame, str(timestamp),
            (0 , h -10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            2)
        fps.update()

        # show the output frame
        # cv2.imshow("Frame", frame)

        with lock:
            FrameTimeFormat = "Time spent on Frame {} : {}us "
            frameID = frameID + 1
            print(FrameTimeFormat.format(frameID , (datetime.datetime.now() - timestamp).microseconds ))
            outputFrame = frame.copy()





    print("Quitting\n")

        # update the FPS counter


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                 help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                 help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                 help="# of frames used to construct the background model")
    ap.add_argument("-p", "--prototxt", required=True,
                 help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                 help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=objectDetection, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,threaded=True, use_reloader=False)

    # release the video stream pointer

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    vs.stop()
