#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Handling video."""

# Imports.
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject
from videoStream import VideoStream
from calibrate import modifyCameraIntrinsics
import os
import h5py
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger('capt')

class VideoThreadSignals(QObject):
    """Video signals."""

    # Signals enabling to update application from thread.
    updatePrepFrame = pyqtSignal(np.ndarray, dict, dict) # Update preprocessed frame (after undistort / stereo).
    calibrationDone = pyqtSignal(int, bool, dict)

class VideoThread(QRunnable): # QThreadPool must be used with QRunnable (NOT QThread).
    """Thread handling video."""

    def __init__(self, vidID, args, vision3D):
        """Initialisation."""

        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._args['videoID'] = vidID
        self._needCalibration = False
        self._args['roiCam'] = False # Initialise ROI (raw mode).
        self._run = True
        self._vid = VideoStream(self._args)
        self.vidID = vidID
        self.signals = VideoThreadSignals()

        # Event subscribe.
        vision3D.signals.changeParam.connect(self.onParameterChanged)
        vision3D.signals.stop.connect(self.stop)

        # Get camera calibration parameters from target camera.
        self._cal = {}
        fileID = f"{args['videoType']}{vidID}"
        calibType = 'fsh' if args['fisheye'] else 'std'
        fname = f"{fileID}-{calibType}.h5"
        if os.path.isfile(fname):
            fdh = h5py.File(fname, 'r')
            for key in fdh:
                self._cal[key] = fdh[key][...]
            fdh.close()
        assert len(self._cal.keys()) > 0, f"camera {vidID} is not calibrated."

        # Set up info/debug log on demand.
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

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
        fileIDStr = f"{args['videoType']}{vidIDStr}"
        fnameStr = f"{fileIDStr}-{calibType}.h5"
        if os.path.isfile(fnameStr):
            fdh = h5py.File(fnameStr, 'r')
            for key in fdh:
                self._stereo[key] = fdh[key][...]
            fdh.close()
        assert len(self._stereo.keys()) > 0, f"camera {vidIDStr} is not calibrated."
        msg = f"[stream{self._args['videoID']:02d}-ini]"
        msg += f" side {self._cal['side']}-{vidID:02d} (stereo {self._stereo['side']}-{vidIDStr:02d})"
        msg += f", file {fname} (stereo {fnameStr})"
        logger.info(msg)

    def onParameterChanged(self, param, objType, value):
        """Callback triggered on parameter change."""

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

        # High impact is needed.
        # Keep track of what must be done in order to handle it in main thread (run).
        # Do NOT run job here (too many callbacks may overflow the main thread).
        highImpact = param == 'mode'
        highImpact = highImpact or param == 'alpha' or param == 'fovScale' or param == 'balance'
        highImpact = highImpact or param == 'CAL'
        if highImpact: # Parameters with high impact (time, calibration computation).
            self._needCalibration = (param, newValue)
        else: # Parameters which with no impact (immediate: ROI, DBGcapt).
            self._args[param] = newValue

        # Update logger level.
        if self._args['DBGcapt']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Update GUI.
        if not self._needCalibration:
            self._emitCalibrationDoneSignal() # Enable GUI (no impact).

    def run(self):
        """Run."""

        # Retrieve constant calibration parameters.
        mtx, dist, fps = self._cal['mtx'], self._cal['dist'], 0

        # Run thread.
        while self._run and self._vid.isOpened():
            # Debug on demand.
            if self._args['DBGcapt']:
                msg = f"[stream{self._args['videoID']:02d}-run]"
                msg += f" FPS {fps:02d}"
                msg += self._generateMessage()
                logger.debug(msg)

            # Take actions.
            if self._needCalibration:
                self._runCalibration()
            else:
                fps = self._runCapture(mtx, dist)
        self._vid.release()

    def stop(self):
        """Stop."""

        # Stop thread.
        self._run = False

    def _calibrate(self):
        """Calibrate."""

        # Check if calibration is needed:
        if self._args['mode'] == 'raw':
            self._args['roiCam'] = False # ROI has no meaning here.
            return # Nothing to do.

        # Calibrate camera individually based on free scaling parameter.
        mtx, dist = self._cal['mtx'], self._cal['dist']
        newCamMtx, roiCam = mtx, False
        alpha = self._args['alpha']
        if alpha >= 0.:
            shape = self._cal['shape']
            newCamMtx, roiCam = modifyCameraIntrinsics(self._args, mtx, dist, shape)
        self._args['newCamMtx'] = newCamMtx
        self._args['roiCam'] = roiCam

        # Check if calibration is complete:
        if self._args['mode'] == 'und':
            return # We are done, no need for stereo.

        # Calibrate stereo (sister) camera individually based on free scaling parameter.
        mtxStr, distStr = self._stereo['mtx'], self._stereo['dist']
        newCamMtxStr = mtxStr
        alpha = self._args['alpha']
        if alpha >= 0.:
            shapeStr = self._stereo['shape']
            newCamMtxStr, _ = modifyCameraIntrinsics(self._args, mtxStr, distStr, shapeStr)

        # Get left/right sides.
        imgL, imgR = None, None
        mtxL, mtxR, newCamMtxL, newCamMtxR, distL, distR = None, None, None, None, None, None
        shapeL, shapeR = None, None
        if self._cal['side'] == 'left':
            imgL = self._cal['img']
            mtxL = mtx
            newCamMtxL = newCamMtx
            distL = dist
            shapeL = self._cal['shape']
            imgR = self._stereo['img']
            mtxR = mtxStr
            newCamMtxR = newCamMtxStr
            distR = distStr
            shapeR = self._stereo['shape']
        else:
            imgR = self._cal['img']
            mtxR = mtx
            newCamMtxR = newCamMtx
            distR = dist
            shapeR = self._cal['shape']
            imgL = self._stereo['img']
            mtxL = mtxStr
            newCamMtxL = newCamMtxStr
            distL = distStr
            shapeL = self._stereo['shape']

        # Cut-off: we must have the same number of image points at left/right.
        obj = self._cal['obj']
        nbMinImg = min(len(imgL), len(imgR))
        imgL = imgL[:nbMinImg]
        imgR = imgR[:nbMinImg]
        obj = obj[:nbMinImg]

        # Calibrate.
        assert self._args['mode'] == 'str', f"unknown mode {self._args['mode']}."
        if self._args['fisheye']:
            self._calibrateWithFisheyeCalibration(obj,
                                                  imgL, mtxL, newCamMtxL, distL, shapeL,
                                                  imgR, mtxR, newCamMtxR, distR, shapeR)
        else:
            if self._args['CAL']:
                self._calibrateWithCalibration(obj,
                                               imgL, mtxL, newCamMtxL, distL, shapeL,
                                               imgR, mtxR, newCamMtxR, distR, shapeR)
            else:
                self._calibrateWithoutCalibration(imgL, mtxL, newCamMtxL, distL, shapeL,
                                                  imgR, mtxR, newCamMtxR, distR, shapeR)

    def _calibrateWithFisheyeCalibration(self, obj,
                                         imgL, mtxL, newCamMtxL, distL, shapeL,
                                         imgR, mtxR, newCamMtxR, distR, shapeR):
        """Fisheye calibration using calibration parameters."""

        # Stereo calibration of both cameras.
        # Intrinsic camera matrices stay unchanged, but, rotation/translation/essential/fundamental matrices are computed.
        flags = cv2.fisheye.CALIB_FIX_SKEW # Do NOT use cv2.fisheye.CALIB_CHECK_COND.
        flags += cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC # Caution: imperative for good results.
        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        shape = self._cal['shape']
        nbImg, _, cbSize, _ = obj.shape
        imgPtsL = np.zeros((nbImg, 1, cbSize, 2), imgL.dtype) # Resize image points.
        for idx, img in enumerate(imgL):
            imgPtsL[idx, 0, :, :] = img[:, 0, :]
        imgPtsR = np.zeros((nbImg, 1, cbSize, 2), imgR.dtype) # Resize image points.
        for idx, img in enumerate(imgR):
            imgPtsR[idx, 0, :, :] = img[:, 0, :]
        _, newCamMtxL, distL, newCamMtxR, distR, rot, trans = cv2.fisheye.stereoCalibrate(obj, imgPtsL, imgPtsR,
                                                                                          newCamMtxL, distL,
                                                                                          newCamMtxR, distR,
                                                                                          shape,
                                                                                          criteria=criteria,
                                                                                          flags=flags)
        self._args['trans'] = trans

        # Stereo rectification based on calibration.
        flags = 0
        rectL, rectR, prjCamMtxL, prjCamMtxR, _ = cv2.fisheye.stereoRectify(newCamMtxL, distL,
                                                                            newCamMtxR, distR,
                                                                            shape, rot, trans,
                                                                            flags,
                                                                            fov_scale=self._args['fovScale'],
                                                                            balance=self._args['balance'],
                                                                            newImageSize=shape)
        stereoMap = None
        if self._cal['side'] == 'left':
            stereoMap = cv2.fisheye.initUndistortRectifyMap(newCamMtxL, distL, rectL, prjCamMtxL, shapeL, cv2.CV_16SC2)
        else:
            stereoMap = cv2.fisheye.initUndistortRectifyMap(newCamMtxR, distR, rectR, prjCamMtxR, shapeR, cv2.CV_16SC2)
        self._args['roiCam'] = False # With fisheye calibration, no ROI.
        self._args['stereoMap'] = stereoMap

    def _calibrateWithCalibration(self, obj,
                                  imgL, mtxL, newCamMtxL, distL, shapeL,
                                  imgR, mtxR, newCamMtxR, distR, shapeR):
        """Standard calibration using calibration parameters."""

        # Stereo calibration of both cameras.
        # Intrinsic camera matrices stay unchanged, but, rotation/translation/essential/fundamental matrices are computed.
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        shape = self._cal['shape']
        _, newCamMtxL, distL, newCamMtxR, distR, rot, trans, _, _ = cv2.stereoCalibrate(obj, imgL, imgR,
                                                                                        newCamMtxL, distL,
                                                                                        newCamMtxR, distR,
                                                                                        shape,
                                                                                        criteria=criteria,
                                                                                        flags=flags)
        self._args['trans'] = trans

        # Stereo rectification based on calibration.
        alpha = -1 # Default scaling.
        if self._args['alpha'] >= 0.:
            alpha = self._args['alpha']
        rectL, rectR, prjCamMtxL, prjCamMtxR, _, roiCamL, roiCamR = cv2.stereoRectify(newCamMtxL, distL,
                                                                                      newCamMtxR, distR,
                                                                                      shape, rot, trans,
                                                                                      alpha=alpha,
                                                                                      newImageSize=shape)
        stereoMap = None
        if self._cal['side'] == 'left':
            stereoMap = cv2.initUndistortRectifyMap(newCamMtxL, distL, rectL, prjCamMtxL, shapeL, cv2.CV_16SC2)
            self._args['roiCam'] = roiCamL
        else:
            stereoMap = cv2.initUndistortRectifyMap(newCamMtxR, distR, rectR, prjCamMtxR, shapeR, cv2.CV_16SC2)
            self._args['roiCam'] = roiCamR
        self._args['stereoMap'] = stereoMap

    def _calibrateWithoutCalibration(self,
                                     imgL, mtxL, newCamMtxL, distL, shapeL,
                                     imgR, mtxR, newCamMtxR, distR, shapeR):
        """Standard calibration without calibration parameters."""

        # Compute fundamental matrix without knowing intrinsic parameters of the cameras and their relative positions.
        imgPtsL = []
        for img in imgL:
            for corner in img:
                imgPtsL.append(np.array(corner[0]))
        imgPtsL = np.array(imgPtsL)
        imgPtsR = []
        for img in imgR:
            for corner in img:
                imgPtsR.append(np.array(corner[0]))
        imgPtsR = np.array(imgPtsR)
        fMtx, _ = cv2.findFundamentalMat(imgPtsL, imgPtsR, cv2.FM_RANSAC)

        # Stereo rectification without knowing calibration.
        shape = self._cal['shape']
        _, matHL, matHR = cv2.stereoRectifyUncalibrated(imgPtsL, imgPtsR, fMtx, shape)
        rectL = np.dot(np.dot(np.linalg.inv(mtxL), matHL), mtxL)
        rectR = np.dot(np.dot(np.linalg.inv(mtxR), matHR), mtxR)
        stereoMap = None
        if self._cal['side'] == 'left':
            stereoMap = cv2.initUndistortRectifyMap(mtxL, distL, rectL, newCamMtxL, shapeL, cv2.CV_16SC2)
        else:
            stereoMap = cv2.initUndistortRectifyMap(mtxR, distR, rectR, newCamMtxR, shapeR, cv2.CV_16SC2)
        self._args['roiCam'] = False # Without calibration, no ROI.
        self._args['stereoMap'] = stereoMap
        if 'trans' in self._args:
            del self._args['trans']

    def _runCapture(self, mtx, dist):
        """Run capture."""

        # Get frame and process it.
        frameOK, frame, fps = self._vid.read()
        if frameOK:
            # Undistort or stereo on demand.
            start = time.time()
            if self._args['mode'] == 'und':
                newCamMtx = self._args['newCamMtx']
                if self._args['fisheye']:
                    undFrame = cv2.fisheye.undistortImage(frame, mtx, dist, Knew=newCamMtx)
                else:
                    undFrame = cv2.undistort(frame, mtx, dist, None, newCamMtx)
                frame = undFrame # Replace frame with undistorted frame.
            elif self._args['mode'] == 'str':
                stereoMap = self._args['stereoMap']
                stereoMapX, stereoMapY = stereoMap[0], stereoMap[1]
                strFrame = cv2.remap(frame, stereoMapX, stereoMapY, cv2.INTER_LINEAR)
                frame = strFrame # Replace frame with stereo frame.
            stop = time.time()
            self._args['undistortTime'] = stop - start

            # Show only ROI on demand.
            if self._args['ROI'] and self._args['roiCam']:
                xTop, yTop, width, height = self._args['roiCam']
                roiFrame = np.ones(frame.shape, np.uint8) # Black image.
                roiFrame[yTop:yTop+height, xTop:xTop+width] = frame[yTop:yTop+height, xTop:xTop+width] # Add ROI.
                frame = roiFrame # Replace frame with ROI of undistorted frame.

            # Get image back to application.
            start = time.time()
            dct = {'fps': fps, 'side': self._cal['side']}
            params = self._createParams()
            self.signals.updatePrepFrame.emit(frame, dct, params)
            stop = time.time()
            self._args['updatePrepFrameTime'] = stop - start
            self._args['updatePrepFrameSize'] = frame.nbytes
        return fps

    def _runCalibration(self):
        """Run calibration."""

        # Apply change only before new  calibration.
        param, newValue = self._needCalibration
        self._args[param] = newValue

        # Run calibration.
        start = time.time()
        self._calibrate()
        stop = time.time()
        msg = f"[stream{self._args['videoID']:02d}-cal]"
        msg += f" time {stop - start:.6f} s"
        msg += self._generateMessage(dbgRun=True)
        logger.info(msg)
        self._needCalibration = False
        self._emitCalibrationDoneSignal()

    def _emitCalibrationDoneSignal(self):
        """Emit signal when calibration is done."""

        # Emit 'calibration done' signal.
        hasROI = False if self._args['roiCam'] is False else True
        params = self._createParams()
        self.signals.calibrationDone.emit(self._args['videoID'], hasROI, params)

    def _createParams(self):
        """Create parameters."""

        # Create calibration parameters dictionary.
        focX = -1.
        if self._args['mode'] != 'raw' and 'newCamMtx' in self._args:
            focX = self._args['newCamMtx'][0][0] # Focal distance along X-axis.
            focX = np.abs(focX)
        baseline = -1.
        if self._args['mode'] == 'str' and 'trans' in self._args:
            baseline = self._args['trans'][0] # Distance between cameras.
            baseline = np.abs(baseline)
        params = {}
        if self._cal['side'] == 'left':
            params['focXLeft'] = focX
            params['baselineLeft'] = baseline
        else:
            params['focXRight'] = focX
            params['baselineRight'] = baseline

        return params

    def _generateMessage(self, dbgRun=False):
        """Generate message."""

        # Generate message from options.
        msg = ''
        if dbgRun or self._args['DBGrun']:
            msg += f", mode {self._args['mode']}"
            if self._args['fisheye']:
                msg += f", fovScale {self._args['fovScale']:.3f}"
                msg += f", balance {self._args['balance']:.3f}"
            else:
                msg += f", alpha {self._args['alpha']:.3f}"
                msg += f", CAL {self._args['CAL']}"
            msg += f", ROI {self._args['ROI']}"
        if self._args['DBGprof']:
            msg += f", mode {self._args['mode']}"
            if 'undistortTime' in self._args:
                msg += f", undistortTime {self._args['undistortTime']:.3f}"
        if self._args['DBGcomm']:
            msg += ', comm'
            key = 'updatePrepFrameTime'
            msg += f", {key} {self._args[key]:.3f}"
            key = 'updatePrepFrameSize'
            msg += f", {key} {self._args[key]}"

        return msg
