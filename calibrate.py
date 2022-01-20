#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import os
import argparse
from videoStream import VideoStream
import cv2
import numpy as np
import h5py
import glob

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed to calibrate video streams from captured frames.'
    parser = argparse.ArgumentParser(description=dscr)
    parser.add_argument('--hardware', type=str, required=True, metavar='HW',
                        choices=['arm-jetson', 'arm-nanopc', 'x86'],
                        help='select hardware to run on')
    parser.add_argument('--videoID', type=int, required=True, metavar='ID',
                        help='select video stream to capture')
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
    parser.add_argument('--calibration', type=int, nargs=3, metavar=('CX', 'CY', 'SS'),
                        default=[7, 10, 25],
                        help='calibration: chessboard size (CX, CY), SS square size (mm)')
    parser.add_argument('--calibration-load-frames', dest='load', action='store_true',
                        help='load frames necessary for calibration.')
    parser.add_argument('--alpha', type=float, default=0., metavar='A',
                        help='free scaling parameter used to compute camera matrix.')
    parser.add_argument('--fisheye', dest='fisheye', action='store_true',
                        help='use fisheye cameras.')
    parser.add_argument('--fovScale', type=float, default=1., metavar='F',
                        help='FOV used to compute fisheye camera matrix.')
    parser.add_argument('--balance', type=float, default=0., metavar='F',
                        help='balance used to compute fisheye camera matrix.')
    args = parser.parse_args()

    # Convert calibration parameters.
    args.chessboardX = args.calibration[0]
    args.chessboardY = args.calibration[1]
    args.squareSize = args.calibration[2]

    return args

def getFileID(args):
    # Get file identifier.
    fileID = '%s%d'%(args.videoType, args.videoID)
    return fileID

def calibrateCameraCheck(obj, img, rvecs, tvecs, mtx, dist):
    # Check camera calibration.
    meanError = 0
    for idx in range(len(obj)):
        imgPrj, _ = cv2.projectPoints(obj[idx], rvecs[idx], tvecs[idx], mtx, dist)
        error = cv2.norm(img[idx], imgPrj, cv2.NORM_L2)/len(imgPrj)
        meanError += error
    print('    Total error: {}'.format(meanError/len(obj)))
    cv2.destroyAllWindows()
    input('    Press any key to continue...')

def calibrateCameraFisheye(args, obj, img, shape):
    # Fisheye camera calibration.
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    objExp = np.expand_dims(np.asarray(obj), -2)
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objExp, img, shape, mtx, dist)
    fileID = getFileID(args)
    fdh = h5py.File('%s-fsh.h5'%fileID, 'w')
    fdh.create_dataset('ret', data=ret)
    fdh.create_dataset('mtx', data=mtx)
    fdh.create_dataset('dist', data=dist)
    fdh.create_dataset('rvecs', data=rvecs)
    fdh.create_dataset('tvecs', data=tvecs)
    fdh.create_dataset('obj', data=obj)
    fdh.create_dataset('img', data=img)
    fdh.create_dataset('shape', data=shape)
    fdh.close()
    calibrateCameraCheck(obj, img, rvecs, tvecs, mtx, dist)

    return mtx, dist

def calibrateCamera(args, obj, img, shape):
    # Camera calibration.
    ret, mtx, dist, rvecs, tvecs, stdDevInt, stdDevExt, perViewErr = cv2.calibrateCameraExtended(obj, img, shape, None, None)
    fileID = getFileID(args)
    fdh = h5py.File('%s-std.h5'%fileID, 'w')
    fdh.create_dataset('ret', data=ret)
    fdh.create_dataset('mtx', data=mtx)
    fdh.create_dataset('dist', data=dist)
    fdh.create_dataset('rvecs', data=rvecs)
    fdh.create_dataset('tvecs', data=tvecs)
    fdh.create_dataset('obj', data=obj)
    fdh.create_dataset('img', data=img)
    fdh.create_dataset('shape', data=shape)
    fdh.close()
    print('    Std dev int: min', np.min(stdDevInt), 'max', np.max(stdDevInt), 'ave', np.average(stdDevInt))
    print('    Std dev ext: min', np.min(stdDevExt), 'max', np.max(stdDevExt), 'ave', np.average(stdDevExt))
    print('    pView error: min', np.min(perViewErr), 'max', np.max(perViewErr), 'ave', np.average(perViewErr))
    calibrateCameraCheck(obj, img, rvecs, tvecs, mtx, dist)

    return mtx, dist

def chessboardCalibration(args, frame, obj, img, delay=0):
    # Termination criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbX, cbY = args.chessboardX, args.chessboardY
    objPt = None
    if args.fisheye:
        objPt = np.zeros((1, cbX*cbY, 3), np.float32) # Caution: dimension for fisheye/standard are not the same.
        objPt[0, :, :2] = np.mgrid[0:cbX, 0:cbY].T.reshape(-1, 2)
    else:
        objPt = np.zeros((cbX*cbY, 3), np.float32) # Caution: dimension for fisheye/standard are not the same.
        objPt[:, :2] = np.mgrid[0:cbX, 0:cbY].T.reshape(-1, 2)
    objPt = objPt*args.squareSize # This makes 3D points spaced as they really are on the (physical) chessboard.

    # Find the chess board corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cbX, cbY), None)

    # If found, add object points, image points (after refining them)
    nbc = 0 if corners is None else len(corners)
    print(' Chessboard found %s with %d corners' % (ret, nbc), end='', flush=True)
    ret = ret and (nbc == cbX*cbY) # Check all chessboard corners have been found.
    if ret == True:
        # Draw and display the corners
        print(': keep', flush=True)
        cornersSP = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        frameWCC = frame.copy()
        cv2.drawChessboardCorners(frameWCC, (cbX, cbY), cornersSP, ret)
        cv2.imshow('Captured frame with found chessboard corners', frameWCC)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

        # Add corners.
        obj.append(objPt)
        img.append(cornersSP) # Keep corners found with sub pixels.
    else:
        print(': drop (%d corners)' % nbc, flush=True)

    return ret

def initFrames(args):
    # Initialise frames needed for calibration.
    frames = []
    obj = [] # 3d point in real world space
    img = [] # 2d points in image plane.
    shape = None
    print('Loading frames...', flush=True)
    fileID = getFileID(args)
    for fname in sorted(glob.glob(fileID + '-*.jpg')):
        print('  Loading %s...' % fname, end='', flush=True)
        frame = cv2.imread(fname)
        height, width, channel = frame.shape # Numpy shapes are height / width / channel.
        shape = (width, height) # OpenCV shapes are width / height.
        ret = chessboardCalibration(args, frame, obj, img, delay=1000)
        if ret:
            frames.append(frame)

    return frames, obj, img, shape

def initCalibration(args):
    # Initialise calibration parameters if available.
    mtx, dist, newCamMtx = None, None, None
    shape = None
    fileID = getFileID(args)
    calibType = 'fsh' if args.fisheye else 'std'
    fname = '%s-%s.h5'%(fileID, calibType)
    if os.path.isfile(fname):
        fdh = h5py.File(fname, 'r')
        mtx = fdh['mtx'][...]
        dist = fdh['dist'][...]
        shape = fdh['shape'][...]
        fdh.close()

    if args.fisheye:
        newCamMtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, shape,
                                                                           np.eye(3), new_size=shape,
                                                                           fov_scale=args.fovScale,
                                                                           balance=args.balance)
    else:
        alpha = args.alpha
        newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, alpha, shape)

    return mtx, dist, newCamMtx

def runCalibration(args):
    # Calibrate camera.
    mtx, dist, newCamMtx = None, None, None
    frames, obj, img, shape = initFrames(args)
    assert len(frames) > 0, 'no frame, no calibration.'
    print('  Calibrating...', flush=True)
    if args.fisheye:
        mtx, dist = calibrateCameraFisheye(args, obj, img, shape)
        newCamMtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, shape,
                                                                           np.eye(3), new_size=shape,
                                                                           fov_scale=args.fovScale,
                                                                           balance=args.balance)
    else:
        mtx, dist = calibrateCamera(args, obj, img, shape)
        alpha = args.alpha
        newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, alpha, shape)

    return mtx, dist, newCamMtx

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Initialise: load frames and calibrate, or, reuse previous calibration.
    mtx, dist, newCamMtx = None, None, None
    if args.load:
        mtx, dist, newCamMtx = runCalibration(args)
    else:
        mtx, dist, newCamMtx = initCalibration(args)

    # Capture video stream.
    vid = VideoStream(vars(args))
    print('Capturing frames...', flush=True)
    while(vid.isOpened()):
        # Get video frame.
        frameOK, frame, fps = vid.read()
        if not frameOK:
            continue

        # Display the resulting frame.
        print('  FPS %d'%fps, flush=True)
        cv2.imshow('Video raw [q quit]', frame)
        if mtx is not None and dist is not None:
            if args.fisheye:
                undFrame = cv2.fisheye.undistortImage(frame, mtx, dist, Knew=newCamMtx)
                cv2.imshow('Video undistorted [q quit]', undFrame)
            else:
                undFrame = cv2.undistort(frame, mtx, dist, None, newCamMtx)
                cv2.imshow('Video undistorted [q quit]', undFrame)

        # Press 'q' to quit.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to quit.
            print('  Exiting...', flush=True)
            break

    # After the loop release the video stream.
    print('Releasing video stream...', flush=True)
    vid.release()
    # Destroy all the windows.
    print('Destroying windows...', flush=True)
    cv2.destroyAllWindows()

# Main program.
if __name__ == '__main__':
    sys.exit(main())
