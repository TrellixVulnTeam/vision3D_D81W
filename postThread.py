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
                msg += ', numDisparities %d'%self._args['numDisparities']
                msg += ', blockSize %d'%self._args['blockSize']
                logger.debug(msg)

            # Checks.
            if not self._args['depth']:
                continue
            if self._post['left'] is None or self._post['right'] is None:
                continue

            # Get frames to postprocess.
            self._postLock.acquire()
            frameL = cv2.cvtColor(self._post['left'], cv2.COLOR_BGR2GRAY)
            frameR = cv2.cvtColor(self._post['right'], cv2.COLOR_BGR2GRAY)
            self._postLock.release()

            # Postprocess.
            if self._stereo is None:
                self._stereo = cv2.StereoBM_create(numDisparities=self._args['numDisparities'],
                                                   blockSize=self._args['blockSize'])
            disparity = self._stereo.compute(frameL, frameR)
            scaledDisparity = disparity - np.min(disparity)
            if np.max(scaledDisparity) > 0:
                scaledDisparity = scaledDisparity * (255/np.max(scaledDisparity))
            scaledDisparity = scaledDisparity.astype(np.uint8)
            msg = 'depth (range 0-255, mean %03d, std %03d)'%(np.mean(scaledDisparity), np.std(scaledDisparity))
            self.signals.updatePostFrame.emit(scaledDisparity, msg)

    def stop(self):
        # Stop thread.
        self._run = False

    def updatePrepFrame(self, frame, side):
        # Postprocess incoming frame.
        self._postLock.acquire()
        self._post[side] = frame # Refresh frame.
        self._postLock.release()
