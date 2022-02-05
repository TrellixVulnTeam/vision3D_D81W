#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import numpy as np
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject
import threading
import cv2
import logging
import time

logger = logging.getLogger('post')

class PostThreadSignals(QObject):
    # Signals enabling to update application from thread.
    updatePostFrame = pyqtSignal(np.ndarray, str, str) # Update postprocessed frame (depth, ...).

class PostThread(QRunnable): # QThreadPool must be used with QRunnable (NOT QThread).
    def __init__(self, args, vision3D, threadLeft, threadRight):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._run = True
        self._post = {'left': None, 'right': None}
        self._postLock = threading.Lock()
        self.signals = PostThreadSignals()
        self._stereo = None
        self._stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        # Event subscribe.
        threadLeft.signals.updatePrepFrame.connect(self.updatePrepFrame)
        threadRight.signals.updatePrepFrame.connect(self.updatePrepFrame)
        vision3D.signals.changeParam.connect(self.onParameterChanged)
        vision3D.signals.stop.connect(self.stop)

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
                msg += self._generateMessage()
                logger.debug(msg)

            # Checks.
            runPost = self._args['depth']
            runPost = runPost or self._args['keypoints']
            runPost = runPost or self._args['stitch']
            if not runPost:
                continue
            if self._post['left'] is None or self._post['right'] is None:
                continue

            # Get frames to postprocess.
            self._postLock.acquire()
            frameL = self._post['left'].copy()
            frameR = self._post['right'].copy()
            self._postLock.release()

            # Postprocess.
            start = time.time()
            frame, fmt, msg = np.ones(frameL.shape, np.uint8), 'GRAY', ''
            try:
                if self._args['depth']:
                    frame, fmt, msg = self._runDepth(frameL, frameR)
                elif self._args['keypoints']:
                    frame, fmt, msg = self._runKeypoints(frameL, frameR)
                elif self._args['stitch']:
                    frame, fmt, msg = self._runStitch(frameL, frameR)
            except:
                if msg == '': # Otherwise, keep more relevant message.
                    msg = 'OpenCV exception!...'
            stop = time.time()
            self._args['postTime'] = stop - start

            # Get image back to application.
            start = time.time()
            self.signals.updatePostFrame.emit(frame, fmt, msg)
            stop = time.time()
            self._args['updatePostFrameTime'] = stop - start
            self._args['updatePostFrameSize'] = frame.nbytes

    def stop(self):
        # Stop thread.
        self._run = False

    def updatePrepFrame(self, frame, side):
        # Postprocess incoming frame.
        self._postLock.acquire()
        self._post[side] = frame # Refresh frame.
        self._postLock.release()

    def _runDepth(self, frameL, frameR):
        # Convert frames to grayscale.
        grayL = self._convertToGrayScale(frameL)
        grayR = self._convertToGrayScale(frameR)

        # Compute depth map.
        if self._stereo is None:
            self._stereo = cv2.StereoBM_create(numDisparities=self._args['numDisparities'],
                                               blockSize=self._args['blockSize'])
        disparity = self._stereo.compute(grayL, grayR)
        scaledDisparity = disparity - np.min(disparity)
        if np.max(scaledDisparity) > 0:
            scaledDisparity = scaledDisparity * (255/np.max(scaledDisparity))
        frame = scaledDisparity.astype(np.uint8)
        msg = 'depth (range 0-255, mean %03d, std %03d)'%(np.mean(scaledDisparity), np.std(scaledDisparity))

        return frame, 'GRAY', msg

    def _computeKeypoints(self, frameL, frameR):
        # To achieve more accurate results, convert frames to grayscale.
        grayL = self._convertToGrayScale(frameL)
        grayR = self._convertToGrayScale(frameR)

        # Detect keypoints.
        kptMode, nbFeatures, msg = None, self._args['nbFeatures'], ''
        if self._args['kptMode'] == 'ORB':
            kptMode = cv2.ORB_create(nfeatures=nbFeatures)
        elif self._args['kptMode'] == 'SIFT':
            kptMode = cv2.SIFT_create(nfeatures=nbFeatures)
        kptL, dscL = kptMode.detectAndCompute(grayL, None)
        kptR, dscR = kptMode.detectAndCompute(grayR, None)
        if len(kptL) == 0 or len(kptR) == 0:
            msg = 'KO: no keypoint'
            return kptL, kptR, [], msg

        # Match keypoints.
        norm = None
        if self._args['kptMode'] == 'ORB':
            norm = cv2.NORM_HAMMING # Better for ORB.
        elif self._args['kptMode'] == 'SIFT':
            norm = cv2.NORM_L2 # Better for SIFT.
        bf = cv2.BFMatcher(norm, crossCheck=False) # Need crossCheck=False for knnMatch.
        matches = bf.knnMatch(dscL, dscR, k=2) # knnMatch crucial to get 2 matches m1 and m2.
        if len(matches) == 0:
            msg = 'KO: no match'
            return kptL, kptR, matches, msg

        # To keep only strong matches.
        bestMatches = []
        for m1, m2 in matches: # For every descriptor, take closest two matches.
            if m1.distance < 0.6 * m2.distance: # Best match has to be closer than second best.
                bestMatches.append(m1) # Lowe’s ratio test.
        if len(bestMatches) == 0:
            msg = 'KO: no best match'
            return kptL, kptR, bestMatches, msg

        return kptL, kptR, bestMatches, msg

    def _runKeypoints(self, frameL, frameR):
        # Compute keypoints.
        kptL, kptR, bestMatches, msg = self._computeKeypoints(frameL, frameR)
        if len(kptL) == 0 or len(kptR) == 0 or len(bestMatches) == 0:
            frame = np.ones(frameL.shape, np.uint8) # Black image.
            return frame, 'GRAY', msg

        # Draw matches.
        frame = cv2.drawMatches(frameL, kptL, frameR, kptR, bestMatches, None)
        minDist = np.min([match.distance for match in bestMatches])
        data = (self._args['kptMode'], minDist, len(bestMatches))
        msg = '%s keypoints (min distance %.3f, nb best matches %d)'%data

        return frame, 'BGR', msg

    def _runStitch(self, frameL, frameR):
        # Compute keypoints.
        kptL, kptR, bestMatches, msg = self._computeKeypoints(frameL, frameR)

        # Stitch images.
        frame, fmt, msg = None, None, 'stitch'
        minMatchCount = 10 # Need at least 10 points to find homography.
        if len(bestMatches) > minMatchCount:
            # Find homography (RANSAC).
            srcPts = np.float32([kptL[match.queryIdx].pt for match in bestMatches]).reshape(-1, 1, 2)
            dstPts = np.float32([kptR[match.trainIdx].pt for match in bestMatches]).reshape(-1, 1, 2)
            homo, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

            # Warp perspective: change field of view.
            rowsL, colsL = frameL.shape[:2]
            rowsR, colsR = frameR.shape[:2]
            lsPtsL = np.float32([[0, 0], [0, rowsL], [colsL, rowsL], [colsL, 0]]).reshape(-1, 1, 2)
            lsPtsR = np.float32([[0, 0], [0, rowsR], [colsR, rowsR], [colsR, 0]]).reshape(-1, 1, 2)
            lsPtsR = cv2.perspectiveTransform(lsPtsR, homo)

            # Stitch images.
            lsPts = np.concatenate((lsPtsL, lsPtsR), axis=0)
            [xMin, yMin] = np.int32(lsPts.min(axis=0).ravel() - 0.5)
            [xMax, yMax] = np.int32(lsPts.max(axis=0).ravel() + 0.5)
            transDist = [-xMin, -yMin] # Translation distance.
            homoTranslation = np.array([[1, 0, transDist[0]], [0, 1, transDist[1]], [0, 0, 1]])
            frame = cv2.warpPerspective(frameR, homoTranslation.dot(homo), (xMax-xMin, yMax-yMin))
            frame[transDist[1]:rowsL+transDist[1], transDist[0]:colsL+transDist[0]] = frameL
            fmt = 'BGR'
            msg += ' OK with %d best matches'%len(bestMatches)

            # Crop on demand.
            if self._args['crop']:
                frame = self._cropFrame(frame)

            # Resize frame to initial frame size (to avoid huge change of dimensions that may occur).
            frame = cv2.resize(frame, (colsL, rowsR), interpolation = cv2.INTER_LINEAR)
        else:
            frame = np.ones(frameL.shape, np.uint8) # Black image.
            fmt = 'GRAY'
            msg += ' KO, not enough matches found (%d/%d)'%(len(bestMatches), minMatchCount)

        return frame, fmt, msg

    @staticmethod
    def _cropFrame(frame):
        # Add black borders around frame to ease thresholding.
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

        # Thresholding (gray scale).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thrFrame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Find biggest contour in thresholded frame.
        cnts, _ = cv2.findContours(thrFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roiArea = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thrFrame.shape, dtype="uint8")
        xCenter, yCenter, width, height = cv2.boundingRect(roiArea)
        cv2.rectangle(mask, (xCenter, yCenter), (xCenter + width, yCenter + height), 255, -1)

        # Find best Region Of Interest (ROI).
        roiMinRectangle = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            roiMinRectangle = cv2.erode(roiMinRectangle, None)
            sub = cv2.subtract(roiMinRectangle, thrFrame)

        # Get best Region Of Interest (ROI).
        cnts, _ = cv2.findContours(roiMinRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roiArea = max(cnts, key=cv2.contourArea)

        # Crop.
        xCenter, yCenter, width, height = cv2.boundingRect(roiArea)
        frame = frame[yCenter:yCenter + height, xCenter:xCenter + width]

        return frame

    @staticmethod
    def _convertToGrayScale(frame):
        # Convert to gray scale if needed.
        convertToGrayScale = True
        if len(frame.shape) == 3: # RGB, BGR or GRAY.
            if frame.shape[2] == 1: # GRAY.
                convertToGrayScale = False
        else: # GRAY.
            convertToGrayScale = False
        if convertToGrayScale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def _generateMessage(self):
        # Generate message from options.
        msg = ''
        if self._args['DBGrun']:
            msg += ', depth %s'%self._args['depth']
            if self._args['depth']:
                msg += ', numDisparities %d'%self._args['numDisparities']
                msg += ', blockSize %d'%self._args['blockSize']
            msg += ', keypoints %s'%self._args['keypoints']
            if self._args['keypoints']:
                msg += ', kptMode %s'%self._args['kptMode']
                msg += ', nbFeatures %d'%self._args['nbFeatures']
            msg += ', stitch %s'%self._args['stitch']
            if self._args['stitch']:
                msg += ', crop %s'%self._args['crop']
        if self._args['DBGprof']:
            if self._args['depth']:
                msg += ', depth'
            if self._args['keypoints']:
                msg += ', keypoints'
            if self._args['stitch']:
                msg += ', stitch'
            if 'postTime' in self._args:
                msg += ', postTime %.3f'%self._args['postTime']
        if self._args['DBGcomm']:
            msg += ', comm'
            if 'updatePostFrameTime' in self._args:
                msg += ', updatePostFrameTime %.3f'%self._args['updatePostFrameTime']
            if 'updatePostFrameSize' in self._args:
                msg += ', updatePostFrameSize %d'%self._args['updatePostFrameSize']

        return msg
