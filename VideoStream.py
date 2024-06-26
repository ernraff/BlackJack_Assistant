############## Camera video stream creator ###############
#
# Description: Defines the VideoStream object, which controls
# acquisition of frames USB camera. The object uses
# multi-threading to aquire camera frames in a separate thread from the main
# program. This allows the main thread to grab the most recent camera frame
# without having to take it directly from the camera feed, reducing I/O time,
# which slightly improves framerate.
#
# When using this with a USB Camera on a desktop or laptop, the framerate tends
# to be too fast. The Card Detector program still works.
#
# See the following web pages for a full explanation of the source code:
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Import the necessary packages
from threading import Thread
import cv2


class VideoStream:
    """Camera object"""
    def __init__(self, resolution=(640,480),framerate=30,src=0):

        # Create a variable to indicate if it's a USB camera or PiCamera.

        # Initialize the USB camera and the camera image stream
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        #ret = self.stream.set(5,framerate) #Doesn't seem to do anything so it's commented out

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	    # Create a variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread to read frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):

        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
		# Return the most recent frame
        return self.frame

    def stop(self):
		# Indicate that the camera and thread should be stopped
        self.stopped = True
