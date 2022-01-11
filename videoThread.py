#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from videoStream import VideoStream
import os
import h5py
import cv2
import numpy as np
import logging

class VideoThread(QThread):
    # Signals enabling to update application from thread.
    changePixmapSignal = pyqtSignal(np.ndarray, QLabel, int, QLabel)

    def __init__(self, args, imgLbl, txtLbl, vision3D):
        # Initialise.
        super().__init__()
        self._args = args
        self._imgLbl = imgLbl
        self._txtLbl = txtLbl
        vision3D.changeParamSignal.connect(self.onParameterChanged)
        self._run = True
        self._vid = VideoStream(self._args)

        # Get camera calibration parameters if any.
        self._cpr = {}
        self._vidID = VideoStream.getVideoID(args)
        videoName = '%s%d'%(args['videoType'], self._vidID)
        fname = '%s.h5'%videoName
        if os.path.isfile(fname):
            fdh = h5py.File(fname, 'r')
            self._cpr['mtx'] = fdh['mtx'][...]
            self._cpr['dist'] = fdh['dist'][...]
            fdh.close()
        assert len(self._cpr.keys()) > 0, 'camera %d is not calibrated'%self._vidID

        # Set up alpha.
        self._onAlphaChanged()

        # Set up debug logging on demand.
        if self._args['debug']:
            logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    @pyqtSlot(str, str, object)
    def onParameterChanged(self, param, objType, value):
        # Convert parameter (type cast) and update args.
        if objType == 'int':
            self._args[param] = int(value)
        elif objType == 'double':
            self._args[param] = float(value)
        elif objType == 'bool':
            self._args[param] = bool(value)
        elif objType == 'str':
            self._args[param] = str(value)
        else:
            assert True, 'unknown type'

        # Take action.
        if param == 'alpha':
            self._onAlphaChanged()

    def _onAlphaChanged(self):
        # Calibrate each camera.
        alpha = self._args['alpha']
        imgSize = (self._vid.width, self._vid.height)
        newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(self._cpr['mtx'], self._cpr['dist'],
                                                          imgSize, alpha, imgSize)
        self._args['newCamMtx'] = newCamMtx
        self._args['roiCam'] = roiCam

    def run(self):
        # Retrieve constant calibration parameters.
        mtx, dist = self._cpr['mtx'], self._cpr['dist']

        # Run thread.
        while self._run and self._vid.isOpened():
            frameOK, frame, fps = self._vid.read()
            if frameOK:
                # Undistort on demand: undistorted frame has the same size and type as the original frame.
                if self._args['mode'] == 'undistort':
                    newCamMtx = self._args['newCamMtx']
                    undFrame = cv2.undistort(frame, mtx, dist, None, newCamMtx)
                    frame = undFrame # Replace frame with undistorted frame.
                    if self._args['ROI']: # Show only ROI on demand.
                        x, y, width, height = self._args['roiCam']
                        roiFrame = np.ones(frame.shape, np.uint8) # Black image.
                        roiFrame[y:y+height, x:x+width] = frame[y:y+height, x:x+width] # Add ROI.
                        frame = roiFrame # Replace frame with ROI of undistorted frame.

                # Debug on demand.
                if self._args['debug']:
                    self._logging(fps)

                # Get image back to application.
                self.changePixmapSignal.emit(frame, self._imgLbl, fps, self._txtLbl)
        self._vid.release()

    def stop(self):
        # Stop thread.
        self._run = False
        self.wait()

    def _logging(self, fps):
        # Log current informations.
        msg = 'stream %d'%self._vidID
        msg += ', FPS %d'%fps
        msg += ', mode %s'%self._args['mode']
        msg += ', alpha %.3f'%self._args['alpha']
        msg += ', ROI %s'%self._args['ROI']
        logging.debug(msg)
