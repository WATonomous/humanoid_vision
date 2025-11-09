
from ultralytics import YOLO
import torch
import torch.nn as nn

# a model we will be using
class Module:
	def __init__(self, name = "Module", description = "Custom model container"):
		self.name = name
		self.description = description
		self.layers = list()
		
	def addLayer(self,layer : Layer):
		# Add a layer to a list
		self.layers.append(layer)
		
	def forward(self,dataIn):
		# run all the data through the layers
		data = dataIn
		for layer in self.layers:
			data = layer(dataIn)
			
		return data
		
	def summary():
		# print a summary of the module, things like name, layers, description, etc
		print()
		
		
class Layer:
	# general layer metadata
	def __init__(self, name = "Layer", description = "Generic model layer"):
		self.name = name
		self.description = description
	
	def process(dataIn):
		# returns a processed version of the data based on what the layer does
		
	
# example layers
class ObjectDetectionLayer(Layer):
	# an input layer
	def process(dataIn):
		# takes an image or a frame and returns bounding boxes
		
		# calls inference pipeline with the model code
		
class OrientationDetectionLayer(Layer):
	def process(dataIn):
		# takes in an image and a bounding box and returns the objects orientation
		
class YOLO(ObjectDetectionLayer):
	# more specific type of object detection based on the YOLO algorithm
	
# etc etc



#--------------------------------------usage example--------------------------------------
""" pipeline = Module("Scene Understanding")

pipeline.addLayer(ObjectDetectionLayer("YOLO"))
pipeline.addLayer(FilteringLayer("Computer Mouse"))
pipeline.addLayer(CoordinateLayer())
pipeline.addLayer(OutputLayer())

result = pipeline.forward(cameraModule.getFrame()) """