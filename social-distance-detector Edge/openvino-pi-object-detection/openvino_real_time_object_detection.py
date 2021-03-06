# USAGE
# python openvino_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import pandas as pd
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-u", "--movidius", type=bool, default=0,
    help="boolean indicating if the Movidius should be used")
ap.add_argument("-i", "--input", help="path to the input video file")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# specify the target device as the Myriad processor on the NCS
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# if a video path was not supplied, grab a reference to the webcam
if args["input"] is None:
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(os.path.abspath(args["input"]))
time.sleep(2.0)
fps = FPS().start()
fps_ = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
dic_detection_frame=[]
dic_detection_number=[]
# loop over the frames from the video stream
while True:
    t1 = time.perf_counter()
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    (grabbed, frame) = vs.read()
    if not grabbed:
        d = {'Frame': dic_detection_frame, 'Detections': dic_detection_number}
        df = pd.DataFrame(data=d)
        df.to_csv("Detecciones_Detection_MobilNetSSD.csv")
        break
    frame = imutils.resize(frame, width=500)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    detectframecount += 1
    dic_detection_frame.append(detectframecount)
    detection_count_person=0

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
            if(idx == 15):
                detection_count_person+= 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(frame, label, (startX, y),
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, fps_, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    dic_detection_number.append(detection_count_person)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        d = {'Frame': dic_detection_frame, 'Detections': dic_detection_number}
        df = pd.DataFrame(data=d)
        df.to_csv("Detecciones_Detection_MobilNetSSD.csv")
        break

    # update the FPS counter
    fps.update()
    # FPS calculation
    framecount += 1
    if framecount >= 15:
        fps_       = "{:.1f} FPS".format(time1/15)
        detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
        framecount = 0
        detectframecount = 0
        time1 = 0
        time2 = 0
    t2 = time.perf_counter()
    elapsedTime = t2-t1
    time1 += 1/elapsedTime
    time2 += elapsedTime
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
