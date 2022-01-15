#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from videoStream import VideoStream
import cv2
import numpy as np
import h5py
import glob

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed to check and calibrate video stream.'
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
    parser.add_argument('--calibration', type=int, nargs=4, metavar=('NF', 'CX', 'CY', 'SS'),
                        default=[10, 7, 10, 25],
                        help='calibration: NF frames, chessboard size (CX, CY), SS square size (mm)')
    parser.add_argument('--calibration-reload-frames', dest='reload', action='store_true',
                        help='reload frames, if any, used during previous calibration.')
    parser.add_argument('--alpha', type=float, default=0., metavar='A',
                        help='free scaling parameter used to compute camera matrix.')
    args = parser.parse_args()

    # Convert calibration parameters.
    args.nbFrames = args.calibration[0]
    args.chessboardX = args.calibration[1]
    args.chessboardY = args.calibration[2]
    args.squareSize = args.calibration[3]

    return args

def calibrateCamera(args, obj, img, shape):
    # Camera calibration.
    ret, mtx, dist, rvecs, tvecs, stdDevInt, stdDevExt, perViewErr = cv2.calibrateCameraExtended(obj, img, shape, None, None)
    videoName = '%s%d'%(args.videoType, args.videoID)
    fdh = h5py.File('%s.h5'%videoName, 'w')
    fdh.create_dataset('ret', data=ret)
    fdh.create_dataset('mtx', data=mtx)
    fdh.create_dataset('dist', data=dist)
    fdh.create_dataset('rvecs', data=rvecs)
    fdh.create_dataset('tvecs', data=tvecs)
    fdh.create_dataset('obj', data=obj)
    fdh.create_dataset('img', data=img)
    height, width = shape # Numpy shapes are height / width.
    shape = (width, height) # OpenCV shapes are width / height.
    fdh.create_dataset('shape', data=shape)
    fdh.close()
    print('    Std dev int: min', np.min(stdDevInt), 'max', np.max(stdDevInt), 'ave', np.average(stdDevInt))
    print('    Std dev ext: min', np.min(stdDevExt), 'max', np.max(stdDevExt), 'ave', np.average(stdDevExt))
    print('    pView error: min', np.min(perViewErr), 'max', np.max(perViewErr), 'ave', np.average(perViewErr))
    meanError = 0
    for idx in range(len(obj)):
        imgPrj, _ = cv2.projectPoints(obj[idx], rvecs[idx], tvecs[idx], mtx, dist)
        error = cv2.norm(img[idx], imgPrj, cv2.NORM_L2)/len(imgPrj)
        meanError += error
    print('    Total error: {}'.format(meanError/len(obj)))

    return mtx, dist

def chessboardCalibration(args, frame, obj, img, delay=0):
    # Termination criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbX, cbY = args.chessboardX, args.chessboardY
    objPt = np.zeros((cbX*cbY, 3), np.float32)
    objPt[:, :2] = np.mgrid[0:cbX, 0:cbY].T.reshape(-1, 2)
    objPt = objPt*args.squareSize # This makes 3D points spaced as they really are on the (physical) chessboard.

    # Find the chess board corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cbX, cbY), None)

    # If found, add object points, image points (after refining them)
    print(' Chessboard found %s' % ret, end='', flush=True)
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
        nbc = 0 if not corners else len(corners)
        print(': drop (%d corners)' % nbc, flush=True)

    return ret

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Data needed for calibration.
    obj = [] # 3d point in real world space
    img = [] # 2d points in image plane.
    shape = None
    frames, videoName = [], '%s%d'%(args.videoType, args.videoID)
    if args.reload: # Reload previous frames on demand.
        print('Loading frames...', flush=True)
        for fname in glob.glob(videoName + '-*.jpg'):
            print('  Loading %s...' % fname, end='', flush=True)
            frame = cv2.imread(fname)
            ret = chessboardCalibration(args, frame, obj, img, delay=1000)
            if ret:
                frames.append(frame)

    # Capture video stream.
    mtx, dist, newCamMtx = None, None, None
    vid = VideoStream(vars(args))
    print('Capturing frames...', flush=True)
    while(vid.isOpened()):
        # Get video frame.
        frameOK, frame, fps = vid.read()
        if not frameOK:
            continue
        height, width, channel = frame.shape
        shape = (height, width) # Numpy shapes are height / width.

        # Display the resulting frame.
        print('  FPS %d'%fps, flush=True)
        cv2.imshow('Video raw [c capture, q quit]', frame)
        if mtx is not None and dist is not None and newCamMtx is not None:
            undFrame = cv2.undistort(frame, mtx, dist, None, newCamMtx)
            cv2.imshow('Video undistorted [q quit]', undFrame)

        # Press 'q' to quit.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to quit.
            print('  Exiting...', flush=True)
            break
        if frames is None:
            continue

        # Wait for key from user according to displayed frame.
        if key == ord('c'): # Press 'c' to capture.
            print('  Capturing frame %s...' % len(frames), end='', flush=True)
            cv2.destroyAllWindows()
            cv2.imshow('Captured frame: keep? [y/n]', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                cv2.destroyAllWindows()
                ret = chessboardCalibration(args, frame, obj, img)
                if ret == True:
                    frames.append(frame)
            cv2.destroyAllWindows()

        # Calibrate camera.
        if args.nbFrames == len(frames):
            print('  Calibrating...', flush=True)
            mtx, dist = calibrateCamera(args, obj, img, shape)
            alpha = args.alpha
            newCamMtx, roiCam = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height),
                                                              alpha, (width, height))

            # Save frames used to calibrate.
            print('  Saving frames...', flush=True)
            for idx, frame in enumerate(frames):
                cv2.imwrite(videoName + '-%02d.jpg'%idx, frame)
            frames = None # Now calibration is done and frames are saved, no more capture.

    # After the loop release the video stream.
    print('Releasing video stream...', flush=True)
    vid.release()
    # Destroy all the windows.
    print('Destroying windows...', flush=True)
    cv2.destroyAllWindows()

# Main program.
if __name__ == '__main__':
    sys.exit(main())
