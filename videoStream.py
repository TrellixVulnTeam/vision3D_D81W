#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import time
import cv2

class VideoStream():
    # Handle a video capture stream.

    def __init__(self, args):
        # Create a video capture stream.
        self._vid, self._nbFrames, self._start = None, 0, time.time()
        vidType = args['videoType']
        vidID = None
        for key, val in args.items():
            if 'videoID' in key:
                vidID = val
        assert vidID is not None, 'no video stream ID.'
        if vidType == 'USB':
            self._vid = cv2.VideoCapture(vidID)
            args['videoCapWidth'] = self._vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            args['videoCapHeight'] = self._vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
            args['videoCapFrameRate'] = self._vid.get(cv2.cv.CV_CAP_PROP_FPS)
        elif vidType == 'CSI':
            cmd = self._gstreamerPipeline(args)
            self._vid = cv2.VideoCapture(cmd%vidID, cv2.CAP_GSTREAMER)
        else:
            assert True, 'create video capture KO, unknown video type.'

    @staticmethod
    def _gstreamerPipeline(args):
        # Get gstreamer pipeline.
        cmd, vidType = None, args['videoType']
        if vidType == 'USB':
            cmd = None # Using gstreamer is not necessary for USB camera.
        elif vidType == 'CSI':
            gst, data, hardware = '', None, args['hardware']
            videoCapWidth, videoCapHeight = args['videoCapWidth'], args['videoCapHeight']
            videoCapFrameRate = args['videoCapFrameRate']
            if hardware == 'arm-jetson': # Nvidia.
                gst = 'nvarguscamerasrc sensor-id=%%d ! '
                gst += 'video/x-raw(memory:NVMM), '
                gst += 'width=(int)%d, height=(int)%d, '
                gst += 'format=(string)NV12, framerate=(fraction)%d/1 ! '
                gst += 'nvvidconv flip-method=%d ! '
                gst += 'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                gst += 'videoconvert ! '
                gst += 'video/x-raw, format=(string)BGR ! appsink'
                videoFlipMethod = args['videoFlipMethod']
                videoDspWidth, videoDspHeight = args['videoDspWidth'], args['videoDspHeight']
                data = (videoCapWidth, videoCapHeight, videoCapFrameRate, videoFlipMethod, videoDspWidth, videoDspHeight)
            elif hardware == 'arm-nanopc': # FriendlyARM.
                gst = 'rkisp device=/dev/video%%d io-mode=1 ! '
                gst += 'video/x-raw, format=NV12, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! '
                gst += 'videoconvert ! appsink'
                data = (videoCapWidth, videoCapHeight, videoCapFrameRate)
            else:
                assert True, 'create video capture KO, unknown hardware.'
            cmd = gst%data
        else:
            assert True, 'create video capture KO, unknown video type.'

        return cmd

    def isOpened(self):
        # Delegate to private data member.
        return self._vid.isOpened()

    def read(self):
        # Delegate to private data member.
        frameOK, frame = self._vid.read()

        # Estimate frame per second.
        self._nbFrames += 1
        fps = int(self._nbFrames/(time.time() - self._start))

        return frameOK, frame, fps

    def release(self):
        # Delegate to private data member.
        self._vid.release()
