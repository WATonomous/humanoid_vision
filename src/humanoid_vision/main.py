"""Placeholder main module.

This module will serve as the entrypoint to run the camera + pipeline and save diffs.
"""

import time
from humanoid_vision.ros_exporter import RosDataStreamer
import numpy as np
import pandas as pd

def main() -> None:
    streamer = RosDataStreamer(topic_name='/data')
    streamer.connect()
    print("Publishing...")
    streamer.publish(np.random.rand(2, 2))
    time.sleep(2)
    streamer.close()
    print("Done.")


if __name__ == "__main__":
    main()
