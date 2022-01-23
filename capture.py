#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
import threading
from videoStream import VideoStream, cmdLineArgsVideoStream
from calibrate import cmdLineArgsCalibrate, chessboardCalibration
import cv2

# Synchronisation barrier.
sync = None

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed to capture frames from video stream for later calibration.'
    parser = argparse.ArgumentParser(description=dscr)
    cmdLineArgsVideoStream(parser, strRightReq=False)
    cmdLineArgsCalibrate(parser)
    parser.add_argument('--startIdx', type=int, default=0, metavar='I',
                        help='define start capture index')
    args = parser.parse_args()

    # Convert calibration parameters.
    args.chessboardX = args.chessboard[0]
    args.chessboardY = args.chessboard[1]
    args.squareSize = args.chessboard[2]

    return vars(args)

class CaptureThread(threading.Thread):
    def __init__(self, args):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._frame = None
        self._idxFrame = self._args['startIdx']
        self._saveLock = threading.Lock()
        self._saveFrame = False
        self.otherThd = None

    def run(self):
        # Capture frames.
        vid = VideoStream(self._args)
        vidID = self._args['videoID']
        while(vid.isOpened()):
            # Get video frame.
            frameOK, frame, fps = vid.read()
            if not frameOK:
                continue

            # Save frame. Show it.
            self._frame = frame # Refresh frame.
            cv2.imshow('stream%02d - Video raw [s save, q quit]'%vidID, frame)

            # Wait for user action.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Press 'q' to quit.
                print('stream%02d: exiting...'%vidID, flush=True)
                break
            if key == ord('s') or self._saveFrame: # Press 's' to save, or, triggered by other.
                if key == ord('s') and self.otherThd is not None:
                    self.otherThd.triggerSave() # Caution: trigger only on press key.
                self.onSave() # Both threads must take the same picture at the same time.
        vid.release()
        cv2.destroyAllWindows()

    def onSave(self):
        # Check VISUALLY if chessboards are PROPERLY found on BOTH frames: MANDATORY for correct stereo calibration.
        self._saveLock.acquire()
        vidID = self._args['videoID']
        print('stream%02d: looking for chessboard...'%vidID, flush=True)
        obj, img = [], []
        ret = chessboardCalibration(self._args, self._frame, obj, img, msg='stream%02d:'%vidID)
        sync.wait() # Wait for all chessboards (from all threads) to be checked.

        # Wait to know if chessboard corners are found properly.
        key = None
        if ret:
            while key != 'y' and key != 'n':
                key = input('stream%02d: keep? [y/n] '%vidID)
        sync.wait() # Wait for all answers (from all threads): keep? drop?
        if key == 'y':
            # Save frame.
            print('stream%02d: saving frame %02d...'%(vidID, self._idxFrame), flush=True)
            fileID = '%s%d'%(self._args['videoType'], self._args['videoID'])
            cv2.imwrite(fileID + '-%02d.jpg'%self._idxFrame, self._frame)
            self._idxFrame += 1
        self._saveFrame = False
        self._saveLock.release()

    def triggerSave(self):
        # Trigger save as soon as possible.
        self._saveLock.acquire()
        self._saveFrame = True
        self._saveLock.release()

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Create threads to handle OpenCV video streams.
    args['videoID'] = args['videoIDLeft']
    mainThd = CaptureThread(args.copy())
    strThd = None
    if args['videoIDRight']:
        args['videoID'] = args['videoIDRight']
        strThd = CaptureThread(args.copy())

    # Create barrier to get threads to do things step-by-step.
    nbThreads = 2 if strThd is not None else 1
    global sync
    sync = threading.Barrier(nbThreads)

    # Create connection between threads.
    if strThd is not None: # Both threads must take the same picture at the same time.
        mainThd.otherThd = strThd
        strThd.otherThd = mainThd

    # Start threads.
    mainThd.start()
    if strThd is not None:
        strThd.start()

    # Wait for threads to be done.
    mainThd.join()
    if strThd is not None:
        strThd.join()

# Main program.
if __name__ == '__main__':
    sys.exit(main())
