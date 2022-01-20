#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
import threading
from videoStream import VideoStream
import cv2

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed to capture frames from video stream for later calibration.'
    parser = argparse.ArgumentParser(description=dscr)
    parser.add_argument('--hardware', type=str, required=True, metavar='HW',
                        choices=['arm-jetson', 'arm-nanopc', 'x86'],
                        help='select hardware to run on')
    parser.add_argument('--videoID', type=int, required=True, metavar='ID',
                        help='select main video stream to capture')
    parser.add_argument('--videoIDStr', type=int, default=False, metavar='ID',
                        help='select stereo video stream to capture')
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
    args = parser.parse_args()

    return args

class VideoThread(threading.Thread):
    def __init__(self, args):
        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._frame = None
        self._idxFrame = 0
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
                self.onSave()
        vid.release()
        cv2.destroyAllWindows()

    def onSave(self):
        # Save frame.
        self._saveLock.acquire()
        vidID = self._args['videoID']
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
    args = vars(args)
    mainThd, strThd = VideoThread(args), None
    if args['videoIDStr']:
        args['videoID'] = args['videoIDStr']
        strThd = VideoThread(args)

    # Create connection between threads.
    if strThd is not None:
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
