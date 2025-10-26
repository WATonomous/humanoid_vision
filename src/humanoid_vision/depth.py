import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

#the intel realsense camera has a rgb camera and a infrared depth camera
#depth camera produces point cloud frames

#--------------------configuration------------------------
#configuring depth and colour streams

pipeline = rs.pipeline()
config= rs.config()

#streams are all the types of data provided by realsense camera
#pixel format, how data from each frame is encoded
config.enable_stream(stream_type = rs.stream.depth, width = 640,height = 480,format = rs.format.z16,framerate = 30)
config.enable_stream(rs.stream.color, 640,480,rs.format.bgr8,30)

profile = pipeline.start(config)

#--------------------define measurement point------------------------
#if our model already returns a midpoint then we wont need this
def getMidpoint(point1,point2):
    #this function takes object and finds midpoint, so that one point can be inputed to find distance
    return (point1[0]+point2[0]) * 0.5 , (point1[1]+point2[1]) * 0.5 #x,y cooridinates


#import model that will be used to process coloured frame to detect objects
model = YOLO("my_model.pt")

#--------------------processing--------------------------------------
try:
    #loop of processing camera frames
    while True:
        #receiving frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()


        #put colour frame into model to get the bounding boxes or whatever marking location of the objects

        results = model(color_frame,stream=True)

        depth_array = []

        for r in results:
            points = r.boxes.xyxy
            x,y = getMidpoint(points[:2],points[2:])

            #get distance from point
            dist = depth_frame.get_distance(x,y)
            depth_array.append(dist)
        
        print(dist)
        cv2.imshow("Stream",frames)
        key = cv2.waitKey(10)

        if key == 27:
                cv2.destroyAllWindows()
                break

#stop streaming
finally:
    pipeline.stop()