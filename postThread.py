#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import numpy as np
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject
import threading
import cv2
import logging

logger = logging.getLogger('post')

class PostThreadSignals(QObject):
    # Signals enabling to update application from thread.
    updatePostFrame = pyqtSignal(np.ndarray, str) # Update postprocessed frame (depth, ...).

class PostThread(QRunnable): # QThreadPool must be used with QRunnable (NOT QThread).
    def __init__(self, args, vision3D, threadLeft, threadRight):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        vision3D.signals.changeParam.connect(self.onParameterChanged)
        vision3D.signals.stop.connect(self.stop)
        threadLeft.signals.updatePrepFrame.connect(self.updatePrepFrame)
        threadRight.signals.updatePrepFrame.connect(self.updatePrepFrame)
        self._run = True
        self._post = {'left': None, 'right': None}
        self._postLock = threading.Lock()
        self.signals = PostThreadSignals()
        self._stereo = None
        self._stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        # Set up info/debug log on demand.
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

    def onParameterChanged(self, param, objType, value):
        # Lots of events may be spawned: check impact is needed.
        newValue = None
        if objType == 'int':
            newValue = int(value)
        elif objType == 'double':
            newValue = float(value)
        elif objType == 'bool':
            newValue = bool(value)
        elif objType == 'str':
            newValue = str(value)
        else:
            assert True, 'unknown type.'
        if self._args[param] == newValue:
            return # Nothing to do.

        # Update logger level.
        if self._args['DBGpost']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Adapt parameter values to allowed values.
        if param == 'numDisparities': # Must be divisible by 16.
            newValue = (newValue//16)*16
        if param == 'blockSize': # Must be odd.
            newValue = (newValue//2)*2+1

        # Reset stereo.
        if param == 'numDisparities' or param == 'blockSize':
            self._stereo = None

        # Apply change.
        self._args[param] = newValue

    def run(self):
        # Execute post-processing.
        while self._run:
            # Debug on demand.
            if self._args['DBGpost']:
                msg = '[post-run]'
                msg += ' depth %s'%self._args['depth']
                if self._args['depth']:
                    msg += ', numDisparities %d'%self._args['numDisparities']
                    msg += ', blockSize %d'%self._args['blockSize']
                msg += ', stitch %s'%self._args['stitch']
                if self._args['stitch'] and 'stitchStatus' in self._args:
                    msg += ', stitchStatus %s'%self._args['stitchStatus']
                logger.debug(msg)

            # Checks.
            if not self._args['depth'] and not self._args['stitch']:
                continue
            if self._post['left'] is None or self._post['right'] is None:
                continue

            # Get frames to postprocess.
            self._postLock.acquire()
            frameL = cv2.cvtColor(self._post['left'], cv2.COLOR_BGR2GRAY)
            frameR = cv2.cvtColor(self._post['right'], cv2.COLOR_BGR2GRAY)
            self._postLock.release()

            # Postprocess.
            frame, msg = np.ones(frameL.shape, np.uint8), ''
            try:
                if self._args['depth']:
                    frame, msg = self._runDepth(frameL, frameR)
                elif self._args['stitch']:
                    frame, msg = self._runStitch(frameL, frameR)
            except:
                if msg == '': # Otherwise, keep more relevant message.
                    msg = 'OpenCV exception!...'
            self.signals.updatePostFrame.emit(frame, msg)

    def stop(self):
        # Stop thread.
        self._run = False

    def updatePrepFrame(self, frame, side):
        # Postprocess incoming frame.
        self._postLock.acquire()
        self._post[side] = frame # Refresh frame.
        self._postLock.release()

    def _runDepth(self, frameL, frameR):
        # Postprocess.
        if self._stereo is None:
            self._stereo = cv2.StereoBM_create(numDisparities=self._args['numDisparities'],
                                               blockSize=self._args['blockSize'])
        disparity = self._stereo.compute(frameL, frameR)
        scaledDisparity = disparity - np.min(disparity)
        if np.max(scaledDisparity) > 0:
            scaledDisparity = scaledDisparity * (255/np.max(scaledDisparity))
        frame = scaledDisparity.astype(np.uint8)
        msg = 'depth (range 0-255, mean %03d, std %03d)'%(np.mean(scaledDisparity), np.std(scaledDisparity))

        return frame, msg

    def _runStitch(self, frameL, frameR):
        # Stitch frames.
        status, frame = self._stitcher.stitch([frameL, frameR])
        self._args['stitchStatus'] = status
        msg = 'stitch'
        if status != cv2.Stitcher_OK:
            frame = np.ones(frameL.shape, np.uint8) # Black image.
            msg += ' KO'
            if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                msg += ': not enough keypoints detected'
            elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
                msg += ': RANSAC homography estimation failed'
            elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
                msg += ': failing to properly estimate camera features'
        else:
            msg += ' OK'

        return frame, msg
