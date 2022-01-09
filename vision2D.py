#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from videoStream import VideoStream
import cv2
import numpy as np
import h5py

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
    parser.add_argument('--videoFrameRate', type=int, default=30, metavar='FR',
                        help='define frame rate')
    parser.add_argument('--videoFlipMethod', type=int, default=0, metavar='FM',
                        help='define flip method')
    parser.add_argument('--videoDspWidth', type=int, default=640, metavar='W',
                        help='define display width')
    parser.add_argument('--videoDspHeight', type=int, default=360, metavar='H',
                        help='define display height')
    parser.add_argument('--calibration', type=int, nargs=3, metavar=('NF', 'CX', 'CY'),
                        default=[10, 7, 10],
                        help='calibration: NF frames, chessboard size (CX, CY)')
    args = parser.parse_args()

    # Convert calibration parameters.
    args.nbFrames = args.calibration[0]
    args.chessboardX = args.calibration[1]
    args.chessboardY = args.calibration[2]

    return args

def calibrateCamera(args, obj, img, gray):
    # Camera calibration.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, gray.shape[::-1], None, None)
    videoName = '%s%d'%(args.videoType, args.videoID)
    fdh = h5py.File('%s.h5'%videoName, 'w')
    fdh.create_dataset('ret', data=ret)
    fdh.create_dataset('mtx', data=mtx)
    fdh.create_dataset('dist', data=dist)
    fdh.create_dataset('rvecs', data=rvecs)
    fdh.create_dataset('tvecs', data=tvecs)
    fdh.close()

def chessboardCalibration(args, frame, obj, img):
    # Termination criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbX, cbY = args.chessboardX, args.chessboardY
    objPt = np.zeros((cbX*cbY, 3), np.float32)
    objPt[:, :2] = np.mgrid[0:cbX, 0:cbY].T.reshape(-1, 2)

    # Find the chess board corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cbY, cbX), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        obj.append(objPt)
        img.append(corners)

        # Draw and display the corners
        cornersSP = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, (cbY, cbX), cornersSP, ret)

    return ret, gray

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Data needed for calibration.
    obj = [] # 3d point in real world space
    img = [] # 2d points in image plane.
    gray, frames = None, []

    # Capture video stream.
    vid = VideoStream(vars(args))
    while(vid.isOpened()):
        # Get video frame.
        frameOK, frame, fps = vid.read()
        if not frameOK:
            continue

        # Display the resulting frame.
        print('FPS %d'%fps, flush=True)
        cv2.imshow('Video [c capture, q quit]', frame)

        # Wait for key from user according to displayed frame.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): # Press 'c' to capture.
            print('  Capturing frame %s...' % len(frames), end=' ', flush=True)
            cv2.destroyAllWindows()
            cv2.imshow('Captured frame: keep? [y/n]', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                cv2.destroyAllWindows()
                ret, gray = chessboardCalibration(args, frame, obj, img)
                print('Kept: chessboard found %s' % ret, flush=True)
                cv2.imshow('Captured frame: chessboard found %s' % ret, frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if ret == True:
                    frames.append(frame)
            else:
                print('Dropped', flush=True)
            cv2.destroyAllWindows()
        elif key == ord('q'): # Press 'q' to quit.
            print('Exiting...', flush=True)
            break

        # Calibrate camera.
        if args.nbFrames == len(frames):
            print('Calibrating...', flush=True)
            calibrateCamera(args, obj, img, gray)
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
