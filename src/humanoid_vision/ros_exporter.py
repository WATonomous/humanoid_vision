import roslibpy
import numpy as np
import pandas as pd


class RosDataStreamer:
    def __init__(
        self,
        host="localhost",
        port=9090,
        topic_name="/data",
        msg_type="std_msgs/Float32MultiArray",
    ):
        self.ros = roslibpy.Ros(host=host, port=port)
        self.topic = roslibpy.Topic(self.ros, topic_name, msg_type)

    def connect(self):
        self.ros.run()
        self.topic.advertise()

    def close(self):
        self.topic.unadvertise()
        self.ros.terminate()

    def publish(self, data):
        if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            arr = np.asarray(data)
            msg = {
                "layout": {"dim": [], "data_offset": 0},
                "data": arr.astype(np.float32).ravel().tolist(),
            }
            self.topic.publish(msg)
        else:
            self.topic.publish({"data": str(data)})
