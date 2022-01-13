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
import time

class VideoThread(QThread):
    # Signals enabling to update application from thread.
    changePixmapSignal = pyqtSignal(np.ndarray, QLabel, int, QLabel)

    def __init__(self, vidID, args, imgLbl, txtLbl, vision3D):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._args['videoID'] = vidID
        self._imgLbl = imgLbl
        self._txtLbl = txtLbl
        self._needCalibration = (None, None)
        vision3D.changeParamSignal.connect(self.onParameterChanged)
        self._run = True
        self._vid = VideoStream(self._args)

        # Get camera calibration parameters from target camera.
        self._cal = {}
        videoName = '%s%d'%(args['videoType'], vidID)
        fname = '%s.h5'%videoName
        if os.path.isfile(fname):
            fdh = h5py.File(fname, 'r')
            for key in fdh:
                self._cal[key] = fdh[key][...]
            fdh.close()
        assert len(self._cal.keys()) > 0, 'camera %d is not calibrated.'%vidID

        # Get camera calibration parameters from the other camera (for stereo).
        self._stereo, vidIDStr = {}, None
        if self._args['videoIDLeft'] == vidID:
            self._cal['side'] = 'left'
            self._stereo['side'] = 'right'
            vidIDStr = self._args['videoIDRight']
        if self._args['videoIDRight'] == vidID:
            self._cal['side'] = 'right'
            self._stereo['side'] = 'left'
            vidIDStr = self._args['videoIDLeft']
        assert vidIDStr is not None, 'can t get other camera.'
        videoName = '%s%d'%(args['videoType'], vidIDStr)
        fname = '%s.h5'%videoName
        if os.path.isfile(fname):
            fdh = h5py.File(fname, 'r')
            for key in fdh:
                self._stereo[key] = fdh[key][...]
            fdh.close()
        assert len(self._stereo.keys()) > 0, 'camera %d is not calibrated.'%vidIDStr

        # Set up debug logging on demand.
        if self._args['debug']:
            logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    @pyqtSlot(str, str, object)
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

        # Impact is needed.
        # Keep track of what must be done in order to handle it in main thread (run).
        # Do NOT run job here (too many callbacks may overflow the main thread).
        if param == 'mode' or param == 'alpha':
            self._needCalibration = (param, newValue)
        elif param == 'ROI':
            self._args[param] = newValue

    def _calibrate(self):
        # Check if calibration is needed:
        if self._args['mode'] == 'raw':
            self._args['roiCam'] = False # ROI has no meaning here.
            return # Nothing to do.

        # Calibrate each camera individually based on free scaling parameter.
        mtx, dist = self._cal['mtx'], self._cal['dist']
        alpha = self._args['alpha']
        height, width = self._cal['shape']
        shape = (width, height)
        newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, alpha, shape)
        self._args['newCamMtx'] = newCamMtx
        self._args['roiCam'] = roiCam

        # Check if calibration is complete:
        if self._args['mode'] == 'und':
            return # We are done, no need for stereo.

        mtxStr, distStr = self._stereo['mtx'], self._stereo['dist']
        heightStr, widthStr = self._stereo['shape']
        shapeStr = (widthStr, heightStr)
        newCamMtxStr, roiCamStr = cv2.getOptimalNewCameraMatrix(mtxStr, distStr, shapeStr, alpha, shapeStr)

        # Stereo calibration of both cameras.
        # Intrinsic camera matrices stay unchanged, but, rotation/translation/essential/fundamental matrices are computed.
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        obj = self._cal['obj']
        imgL, imgR, newCamMtxL, newCamMtxR, distL, distR, shapeL, shapeR = None, None, None, None, None, None, None, None
        if self._cal['side'] == 'left':
            imgL = self._cal['img']
            newCamMtxL = newCamMtx
            distL = dist
            shapeL = self._cal['shape']
            imgR = self._stereo['img']
            newCamMtxR = newCamMtxStr
            distR = distStr
            shapeR = self._stereo['shape']
        else:
            imgR = self._cal['img']
            newCamMtxR = newCamMtx
            distR = dist
            shapeR = self._cal['shape']
            imgL = self._stereo['img']
            newCamMtxL = newCamMtxStr
            distL = distStr
            shapeL = self._stereo['shape']
        shape = self._cal['shape']
        ret, newCamMtxL, distL, newCamMtxR, distR, rot, trans, eMtx, fMtx = cv2.stereoCalibrate(obj, imgL, imgR,
                                                                                                newCamMtxL, distL,
                                                                                                newCamMtxR, distR,
                                                                                                shape, criteria, flags)

        # Stereo rectification.
        rectScale = 1
        rectL, rectR, prjMtxL, prjMtxR, matQ, roiL, roiR= cv2.stereoRectify(newCamMtxL, distL, newCamMtxR, distR,
                                                                            shape, rot, trans, rectScale, (0,0))
        stereoMap = None
        if self._cal['side'] == 'left':
            stereoMap = cv2.initUndistortRectifyMap(newCamMtxL, distL, rectL, prjMtxL, shapeL, cv2.CV_16SC2)
            self._args['roiCam'] = roiL
        else:
            stereoMap = cv2.initUndistortRectifyMap(newCamMtxR, distR, rectR, prjMtxR, shapeR, cv2.CV_16SC2)
            self._args['roiCam'] = roiR
        self._args['stereoMap'] = stereoMap

    def run(self):
        # Retrieve constant calibration parameters.
        mtx, dist, fps = self._cal['mtx'], self._cal['dist'], 0

        # Run thread.
        while self._run and self._vid.isOpened():
            # Debug on demand.
            if self._args['debug']:
                self._logging(fps)

            # Take actions.
            if self._needCalibration:
                self._runCalibration()
            else:
                fps = self._runCapture(mtx, dist)
        self._vid.release()

    def stop(self):
        # Stop thread.
        self._run = False
        self.wait()

    def _logging(self, fps):
        # Log current informations.
        msg = 'stream%d'%self._args['videoID']
        msg += ', FPS %02d'%fps
        msg += ', mode %s'%self._args['mode']
        msg += ', alpha %.3f'%self._args['alpha']
        msg += ', ROI %s'%self._args['ROI']
        logging.debug(msg)

    def _runCapture(self, mtx, dist):
        # Get frame and process it.
        frameOK, frame, fps = self._vid.read()
        if frameOK:
            # Undistort or stereo on demand.
            if self._args['mode'] == 'und':
                newCamMtx = self._args['newCamMtx']
                undFrame = cv2.undistort(frame, mtx, dist, None, newCamMtx)
                frame = undFrame # Replace frame with undistorted frame.
            elif self._args['mode'] == 'str':
                stereoMap = self._args['stereoMap']
                strFrame = cv2.remap(frame, stereoMap[0], stereoMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                frame = strFrame # Replace frame with stereo frame.

            # Show only ROI on demand.
            if self._args['ROI'] and self._args['roiCam']:
                x, y, width, height = self._args['roiCam']
                roiFrame = np.ones(frame.shape, np.uint8) # Black image.
                roiFrame[y:y+height, x:x+width] = frame[y:y+height, x:x+width] # Add ROI.
                frame = roiFrame # Replace frame with ROI of undistorted frame.

            # Get image back to application.
            self.changePixmapSignal.emit(frame, self._imgLbl, fps, self._txtLbl)
        return fps

    def _runCalibration(self):
        # Reset application image.
        shape = (self._vid.height, self._vid.width)
        frame = np.ones(shape, np.uint8) # Black image.
        self.changePixmapSignal.emit(frame, self._imgLbl, 0, self._txtLbl)

        # Apply change only before new  calibration.
        param, newValue = self._needCalibration
        self._args[param] = newValue

        # Run calibration.
        start = time.time()
        self._calibrate()
        stop = time.time()
        if self._args['debug']:
            msg = 'stream%d'%self._args['videoID']
            msg += ', calib time %.6f s'%(stop - start)
            logging.debug(msg)
        self._needCalibration = False
