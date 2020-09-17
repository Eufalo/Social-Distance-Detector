from pyimagesearch_prueba.centroidtracker import CentroidTracker
from pyimagesearch_prueba.trackableobject import TrackableObject
from pyimagesearch_prueba.detection import Detection as dt
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import math
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo",
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-f", "--skip-frames", type=int, default=50,
	help="# of skip frames between detections")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

class Ventana_Principal():
    def __init__(self):
        #with height of the videoPanel
        #self.window_width = self.videoPanel.frameSize().width()
        #self.window_height = self.videoPanel.frameSize().height()
        #start the start video controller
        #self.start_video.clicked.connect(self.controller_start_clicked)
        #start the add lines controller
        #self.addHLine.clicked.connect(self.controller_addHLine)
        #self.addVLine.clicked.connect(self.controller_addVLine)
        #add controller for the table car detection
        #self.tableCarDetect.cellClicked.connect(self.cell_car_Click)
        #add new manual detection controller
        #self.add_ManualDetection.clicked.connect(self.controller_addManualDetection)
        #Inicialice the class detection
        self.detector = dt(args["yolo"])
    def startDetection(self):
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["input"])
        # initialize the total number of frames processed
        totalFrames = 0
        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        W = None
        H = None

        # initialize the video writer/output
        writer = None
        # start the frames per second throughput estimator
        fps = FPS().start()
        # loop over frames from the video stream
        while True:
            # grab the next frame
            self.frame = vs.read()
            self.frame = self.frame[1]

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if args["input"] is not None and self.frame is None:
                break

            # resize the frame to have a maximum width of 700 pixels
            self.frame = imutils.resize(self.frame, width=700)
            # convert the frame from BGR to RGB for dlib
            # increase the acurazy of the traking
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = self.frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                    (W, H), True)

            # We need to skip some frames  (args["skip_frames"])  because
            # the car detected will be comput expensive and "detection it not in seudo-realtime"
            if totalFrames % args["skip_frames"] == 0:
                #flag true detection active
                self.detector.new_detection(self.frame,True,W, H,rgb,args["confidence"],totalFrames)
            else:
                #flag false tracking active
                self.detector.new_detection(self.frame,False,W, H,rgb,args["confidence"],totalFrames)
            #if self.flagLine.checkState()==2:
            self.detector.paint_linesH(self.frame)
            self.detector.paint_linesV(self.frame)
            #if self.flagBoxes.checkState()==2:
            self.detector.print_boundingbox(self.frame)

            self.detector.pritn_trakingobjct(self.frame)

            # update the car counting information we will be displaying on the
            # the window
            #self.CarCount.display(self.detector.totalCount)

            #if totalFrames % 5 ==0:
                #update the tables
                #self.update_tables()

            # check to see if we should write the self.frame to disk
            #if writer is not None:
                #writer.write(self.frame)

            # show the output frame
            cv2.imshow("Frame", self.frame)
            #self.update_frame()

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
                #self.start_video.setEnabled(True)
            # if the `s` key was pressed, can add special region who cointain a vehicle
            elif key == ord("s"):
                # select the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)
                box = cv2.selectROI("Frame", self.frame, fromCenter=False,
                    showCrosshair=True)
                # create a new object tracker for the bounding box and add it
                # to our multi-object tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle( box[0], box[1], box[0] + box[2], box[1] + box[3])
                tracker.start_track(rgb, rect)
                # add the tracker to our list of self.trackers so we can
                # utilize it during skip frames
                self.detector.trackers.append(tracker)
            # if the `v` key was pressed, u can add a vertical line
            elif key == ord("v"):
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    box = cv2.selectROI("Frame", self.frame, fromCenter=False,
                        showCrosshair=True)
                    # create a new line for counting in that WAy
                    #print((box[0], box[1],box[0]+box[2],box[1]+ box[3]))
                    self.detector.linesV.append((box[0], box[1],box[0]+box[2],box[1]+ box[3]))

            # if the `h` key was pressed, u can add a horizontal line
            elif key == ord("h"):
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    box = cv2.selectROI("Frame", self.frame, fromCenter=False,
                        showCrosshair=True)
                    # create a new line for counting in that WAy
                    self.detector.linesH.append((box[0], box[1]+ box[3],box[0]+box[2],box[1]))
            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # if we are not using a video file, stop the camera video stream
        if not args.get("input", False):
            vs.stop()

        # otherwise, release the video file pointer
        else:
            vs.release()

        # close any open windows
        cv2.destroyAllWindows()
    def cell_car_Click(self):
        #print(self.detector.trackableObjects)
        #self.tableCarDetect.getItem(
        row=self.tableCarDetect.currentItem().row()
        #get the id of the object in the table
        objid=self.tableCarDetect.item(row,0).text()
        self.detector.trackableObjects.get(int(objid)).activeTraking()
        self.detector.pritn_trakingobjct(self.frame)
        self.update_frame()
    def controller_addVLine(self):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", self.frame, fromCenter=False,
            showCrosshair=True)
        # create a new line for counting in that WAy
        self.detector.linesV.append((box[0], box[1],box[0]+box[2],box[1]+ box[3]))
    def controller_addHLine(self):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", self.frame, fromCenter=False,
            showCrosshair=True)

        # create a new line for counting in that WAy
        self.detector.linesH.append((box[0], box[1]+ box[3],box[0]+box[2],box[1]))
    def controller_addManualDetection(self):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", self.frame, fromCenter=False,
            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle( box[0], box[1], box[0] + box[2], box[1] + box[3])
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        tracker.start_track(rgb, rect)
        # add the tracker to our list of self.trackers so we can
        # utilize it during skip frames
        self.detector.trackers.append(tracker)


_ventana = Ventana_Principal()
_ventana.startDetection()
