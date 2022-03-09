#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Handling video stream."""

# Imports.
import time
import cv2

def cmdLineArgsVideoStream(parser, stereo=True, strLeftReq=True, strRightReq=True):
    """Manage command line arguments related to video stream."""

    # Add parser command options related to the video stream.
    if stereo:
        parser.add_argument('--videoIDLeft', type=int, required=strLeftReq, metavar='ID',
                            help='select left video stream to capture')
        parser.add_argument('--videoIDRight', type=int, required=strRightReq, metavar='ID',
                            help='select right video stream to capture')
    else:
        parser.add_argument('--videoID', type=int, required=True, metavar='ID',
                            help='select video stream to capture')
    parser.add_argument('--hardware', type=str, required=True, metavar='HW',
                        choices=['arm-jetson', 'arm-nanopc', 'x86'],
                        help='select hardware to run on')
    parser.add_argument('--videoType', type=str, default='CSI', metavar='T',
                        choices=['CSI', 'USB'],
                        help='select video type')
    parser.add_argument('--videoCapWidth', type=int, default=640, metavar='W',
                        help='define capture width')
    parser.add_argument('--videoCapHeight', type=int, default=360, metavar='H',
                        help='define capture height')
    parser.add_argument('--videoCapFrameRate', type=int, default=30, metavar='FR',
                        help='define capture frame rate')
    parser.add_argument('--videoFlipMethod', type=int, default=0, metavar='FM',
                        help='define flip method')
    parser.add_argument('--videoDspWidth', type=int, default=640, metavar='W',
                        help='define display width')
    parser.add_argument('--videoDspHeight', type=int, default=360, metavar='H',
                        help='define display height')

class VideoStream():
    """Handling video capture stream."""

    def __init__(self, args):
        """Initialisation."""

        # Create a video capture stream.
        self._vid, self._nbFrames, self._start = None, 0, time.time()
        vidType = args['videoType']
        vidID = args['videoID']
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
        self.width = args['videoCapWidth']
        self.height = args['videoCapHeight']

    def isOpened(self):
        """Check if the stream is opened."""

        # Delegate to private data member.
        return self._vid.isOpened()

    def read(self):
        """Read stream."""

        # Delegate to private data member.
        frameOK, frame = self._vid.read()

        # Estimate frame per second.
        self._nbFrames += 1
        fps = int(self._nbFrames/(time.time() - self._start))

        return frameOK, frame, fps

    def release(self):
        """Release stream."""

        # Delegate to private data member.
        self._vid.release()

    @staticmethod
    def _gstreamerPipeline(args):
        """Get gstreamer pipeline."""

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
