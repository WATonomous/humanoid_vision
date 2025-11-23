import roslibpy

ros = roslibpy.Ros(host="localhost", port=9090)
ros.run()
print("Connected: ", ros.is_connected)
ros.terminate()
