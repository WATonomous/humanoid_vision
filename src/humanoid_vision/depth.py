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

#alignment object
align = rs.align(rs.stream.color)

#--------------------define measurement point------------------------
#finds midpoint of given the coordinates of bounding box so that depth can be calculated
def getMidpoint(point1,point2):
    #this function takes object and finds midpoint, so that one point can be inputed to find distance
    return int((point1[0]+point2[0]) * 0.5) , int((point1[1]+point2[1]) * 0.5) #x,y cooridinates


#import some model to detect objects
model = YOLO("yolov8n.pt")
#net = cv2.dnnreadNetFromCaffe("Resources/MobileNetSSD_deploy.prototxt","Resources/MobileNetSSD_deploy.caffemodel")

depth_array=[]
#--------------------processing--------------------------------------
try:
    #loop of processing camera frames
    while True:
        #receiving frames
        frames = pipeline.wait_for_frames()

        #align depth to color
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        #convert frames to np arrays before putting into model
        color_frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        #put colour frame into model to get the bounding boxes or whatever marking location of the objects

        results = model(color_frame,stream=True)

        object_depths = []
        for r in results:
            boxes = r.boxes
            for i,box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = box.conf[0]
                cls_name = model.names[cls]
                if conf>0.5:
                    x1,y1,x2,y2 = box.xyxy[0].numpy()
                    x,y = getMidpoint((x1,y1),(x2,y2))
                    #get distance from point
                    dist = depth_frame.get_distance(x,y)

                    object_depths.append({'id':i,'position':(x,y),'distance':dist})
                    cv2.rectangle(color_frame,
                                  (int(x1),int(y1)),
                                  (int(x2),int(y2)),
                                  (0,255,0),
                                  2)
                    label = f"{cls_name}: {dist:.2f}"
                    cv2.putText(color_frame,
                                label,
                                (int(x1),int(y1)-10),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (0,255,0),
                                2)

        print(object_depths)#printing out information

        cv2.imshow("Depth detection",color_frame) #replace with annotated frame after
        key = cv2.waitKey(10)

        #leaves when escape is pressed
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()