import numpy as np
import os
import cv2


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=True):

        framearray = []

        for image_files in os.listdir(filename):
        	
        	img_path = os.path.join(filename, image_files)

        	img = cv2.imread(img_path)       	
        	
        	img = cv2.resize(img, (self.height, self.width))

        	framearray.append(img)
   
        return np.array(framearray)

    def get_UCF_classname(self, filename):
        x =  filename[filename.find('_') + 1:filename.find('_', 2)]
        return x