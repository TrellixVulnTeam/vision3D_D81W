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
import time
import logging

logger = logging.getLogger()

class VideoThread(QThread):
    # Signals enabling to update application from thread.
    changePixmapSignal = pyqtSignal(np.ndarray, QLabel, int, QLabel)
    calibrationDoneSignal = pyqtSignal(int, bool)

    def __init__(self, vidID, args, imgLbl, txtLbl, vision3D):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._args['videoID'] = vidID
        self._imgLbl = imgLbl
        self._txtLbl = txtLbl
        self._needCalibration = False
        self._args['roiCam'] = False # Initialise ROI (raw mode).
        vision3D.changeParamSignal.connect(self.onParameterChanged)
        self._run = True
        self._vid = VideoStream(self._args)
        self.vidID = vidID

        # Get camera calibration parameters from target camera.
        self._cal = {}
        fileID = '%s%d'%(args['videoType'], vidID)
        calibType = 'fsh' if args['fisheye'] else 'std'
        fname = '%s-%s.h5'%(fileID, calibType)
        if os.path.isfile(fname):
            fdh = h5py.File(fname, 'r')
            for key in fdh:
                self._cal[key] = fdh[key][...]
            fdh.close()
        assert len(self._cal.keys()) > 0, 'camera %d is not calibrated.'%vidID

        # Set up info/debug log on demand.
        logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

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
        fileIDStr = '%s%d'%(args['videoType'], vidIDStr)
        fnameStr = '%s-%s.h5'%(fileIDStr, calibType)
        if os.path.isfile(fnameStr):
            fdh = h5py.File(fnameStr, 'r')
            for key in fdh:
                self._stereo[key] = fdh[key][...]
            fdh.close()
        assert len(self._stereo.keys()) > 0, 'camera %d is not calibrated.'%vidIDStr
        msg = 'stream%02d-ini'%self._args['videoID']
        msg += ', side %s-%02d (stereo %s-%02d)'%(self._cal['side'], vidID, self._stereo['side'], vidIDStr)
        msg += ', file %s (stereo %s)'%(fname, fnameStr)
        logger.info(msg)

        # Set up YOLO.
        self._setupYOLO()

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

        # High impact is needed.
        # Keep track of what must be done in order to handle it in main thread (run).
        # Do NOT run job here (too many callbacks may overflow the main thread).
        highImpact = param == 'mode'
        highImpact = highImpact or param == 'alpha' or param == 'fovScale' or param == 'balance'
        highImpact = highImpact or param == 'CAL'
        if highImpact: # Parameters with high impact (time, calibration computation).
            self._needCalibration = (param, newValue)
        else: # Parameters which with no impact (immediate: ROI, DBG).
            self._args[param] = newValue

        # Update logger level.
        if self._args['DBG']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Update GUI.
        if not self._needCalibration:
            self._emitCalibrationDoneSignal() # Enable GUI (no impact).

    def _calibrate(self):
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
            if self._args['fisheye']:
                newCamMtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, shape,
                                                                                   np.eye(3), new_size=shape,
                                                                                   fov_scale=self._args['fovScale'],
                                                                                   balance=self._args['balance'])
            else:
                newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, alpha, shape)
        self._args['newCamMtx'] = newCamMtx
        self._args['roiCam'] = roiCam

        # Check if calibration is complete:
        if self._args['mode'] == 'und':
            return # We are done, no need for stereo.

        # Calibrate stereo (sister) camera individually based on free scaling parameter.
        mtxStr, distStr = self._stereo['mtx'], self._stereo['dist']
        newCamMtxStr, roiCamStr = mtxStr, False
        alpha = self._args['alpha']
        if alpha >= 0.:
            shapeStr = self._stereo['shape']
            if self._args['fisheye']:
                newCamMtxStr = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtxStr, distStr, shapeStr,
                                                                                      np.eye(3), new_size=shapeStr,
                                                                                      fov_scale=self._args['fovScale'],
                                                                                      balance=self._args['balance'])
            else:
                newCamMtxStr, roiCamStr = cv2.getOptimalNewCameraMatrix(mtxStr, distStr, shapeStr, alpha, shapeStr)

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
        assert self._args['mode'] == 'str', 'unknown mode %s.'%self._args['mode']
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
        ret, newCamMtxL, distL, newCamMtxR, distR, rot, trans = cv2.fisheye.stereoCalibrate(obj, imgPtsL, imgPtsR,
                                                                                            newCamMtxL, distL,
                                                                                            newCamMtxR, distR,
                                                                                            shape,
                                                                                            criteria=criteria,
                                                                                            flags=flags)

        # Stereo rectification based on calibration.
        flags = 0
        rectL, rectR, prjCamMtxL, prjCamMtxR, matQ = cv2.fisheye.stereoRectify(newCamMtxL, distL,
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
        # Stereo calibration of both cameras.
        # Intrinsic camera matrices stay unchanged, but, rotation/translation/essential/fundamental matrices are computed.
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        shape = self._cal['shape']
        ret, newCamMtxL, distL, newCamMtxR, distR, rot, trans, eMtx, fMtx = cv2.stereoCalibrate(obj, imgL, imgR,
                                                                                                newCamMtxL, distL,
                                                                                                newCamMtxR, distR,
                                                                                                shape,
                                                                                                criteria=criteria,
                                                                                                flags=flags)

        # Stereo rectification based on calibration.
        alpha = -1 # Default scaling.
        if self._args['alpha'] >= 0.:
            alpha = self._args['alpha']
        rectL, rectR, prjCamMtxL, prjCamMtxR, matQ, roiCamL, roiCamR = cv2.stereoRectify(newCamMtxL, distL,
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
        fMtx, mask = cv2.findFundamentalMat(imgPtsL, imgPtsR, cv2.FM_RANSAC)

        # Stereo rectification without knowing calibration.
        shape = self._cal['shape']
        ret, matHL, matHR = cv2.stereoRectifyUncalibrated(imgPtsL, imgPtsR, fMtx, shape)
        rectL = np.dot(np.dot(np.linalg.inv(mtxL), matHL), mtxL)
        rectR = np.dot(np.dot(np.linalg.inv(mtxR), matHR), mtxR)
        stereoMap = None
        if self._cal['side'] == 'left':
            stereoMap = cv2.initUndistortRectifyMap(mtxL, distL, rectL, newCamMtxL, shapeL, cv2.CV_16SC2)
        else:
            stereoMap = cv2.initUndistortRectifyMap(mtxR, distR, rectR, newCamMtxR, shapeR, cv2.CV_16SC2)
        self._args['roiCam'] = False # Without calibration, no ROI.
        self._args['stereoMap'] = stereoMap

    def run(self):
        # Retrieve constant calibration parameters.
        mtx, dist, fps = self._cal['mtx'], self._cal['dist'], 0

        # Run thread.
        while self._run and self._vid.isOpened():
            # Debug on demand.
            if self._args['DBG']:
                msg = 'stream%02d-run'%self._args['videoID']
                msg += ', FPS %02d'%fps
                msg += self._generateMessage()
                logger.debug(msg)

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

    def _runCapture(self, mtx, dist):
        # Get frame and process it.
        frameOK, frame, fps = self._vid.read()
        if frameOK:
            # Undistort or stereo on demand.
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

            # Show only ROI on demand.
            if self._args['ROI'] and self._args['roiCam']:
                x, y, width, height = self._args['roiCam']
                roiFrame = np.ones(frame.shape, np.uint8) # Black image.
                roiFrame[y:y+height, x:x+width] = frame[y:y+height, x:x+width] # Add ROI.
                frame = roiFrame # Replace frame with ROI of undistorted frame.

            # Run YOLO on demand.
            if self._args['YOLO']:
                start = time.time()
                self._runYOLODetection(frame)
                stop = time.time()
                self._args['YOLO-time'] = stop - start

            # Get image back to application.
            self.changePixmapSignal.emit(frame, self._imgLbl, fps, self._txtLbl)
        return fps

    def _runCalibration(self):
        # Apply change only before new  calibration.
        param, newValue = self._needCalibration
        self._args[param] = newValue

        # Run calibration.
        start = time.time()
        self._calibrate()
        stop = time.time()
        msg = 'stream%02d-cal'%self._args['videoID']
        msg += self._generateMessage()
        msg += ', time %.6f s'%(stop - start)
        logger.info(msg)
        self._needCalibration = False
        self._emitCalibrationDoneSignal()

    def _emitCalibrationDoneSignal(self):
        # Emit 'calibration done' signal.
        hasROI = False if self._args['roiCam'] is False else True
        self.calibrationDoneSignal.emit(self._args['videoID'], hasROI)

    def _generateMessage(self):
        # Generate message from options.
        msg = ', mode %s'%self._args['mode']
        if self._args['fisheye']:
            msg += ', fovScale %.3f'%self._args['fovScale']
            msg += ', balance %.3f'%self._args['balance']
        else:
            msg += ', alpha %.3f'%self._args['alpha']
            msg += ', CAL %s'%self._args['CAL']
        msg += ', ROI %s'%self._args['ROI']
        msg += ', YOLO %s'%self._args['YOLO']
        if self._args['YOLO']:
            msg += ', confidence %.3f'%self._args['confidence']
            msg += ', nms %.3f'%self._args['nms']
            if 'YOLO-time' in self._args:
                msg += ', YOLO-time %.3f'%self._args['YOLO-time']

        return msg

    def _setupYOLO(self):
        # Load the COCO class labels our YOLO model was trained on.
        labels = open('coco.names').read().strip().split("\n")

        # Initialize a list of colors to represent each possible class label.
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        # load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[idx - 1] for idx in net.getUnconnectedOutLayers()]

        self._yolo = {}
        self._yolo['colors'] = colors
        self._yolo['labels'] = labels
        self._yolo['net'] = net
        self._yolo['ln'] = ln

    def _runYOLODetection(self, frame):
        # Construct a blob from the input frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector.
        net, ln = self._yolo['net'], self._yolo['ln']
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Initialize our lists of detected bounding boxes, confidences, and class IDs.
        boxes, confidences, classIDs = [], [], []

        # Loop over each of the layer outputs.
        height, width = frame.shape[:2]
        for output in layerOutputs:
            # Loop over each of the detections.
            for detection in output:
                # Extract the class ID and confidence (i.e., probability) of the current detection.
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability.
                if confidence > self._args['confidence']:
                    # Scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                    # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height.
                    box = detection[0:4] * np.array([width, height, width, height])
                    boxCenterX, boxCenterY, boxWidth, boxHeight = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and and left corner of the bounding box.
                    boxCenterX = int(boxCenterX - (boxWidth / 2))
                    boxCenterY = int(boxCenterY - (boxHeight / 2))

                    # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([boxCenterX, boxCenterY, int(boxWidth), int(boxHeight)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes.
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self._args['confidence'], self._args['nms'])

        # Ensure at least one detection exists.
        colors, labels = self._yolo['colors'], self._yolo['labels']
        if len(idxs) > 0:
            # Loop over the indexes we are keeping.
            for idx in idxs.flatten():
                # Extract the bounding box coordinates.
                (boxCenterX, boxCenterY) = (boxes[idx][0], boxes[idx][1])
                (boxWidth, boxHeight) = (boxes[idx][2], boxes[idx][3])

                # Draw a bounding box rectangle and label on the frame.
                color = [int(clr) for clr in colors[classIDs[idx]]]
                cv2.rectangle(frame, (boxCenterX, boxCenterY), (boxCenterX + boxWidth, boxCenterY + boxHeight), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[idx]], confidences[idx])
                cv2.putText(frame, text, (boxCenterX, boxCenterY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
