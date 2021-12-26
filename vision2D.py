#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from videoStream import VideoStream
import cv2

def cmdLineArgs():
    # Create parser.
    parser = argparse.ArgumentParser(description='script designed to check video stream.')
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
    args = parser.parse_args()

    return args

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Capture video stream.
    vid = VideoStream(vars(args))
    while(vid.isOpened()):
        # Get video frame.
        frameOK, frame, fps = vid.read()
        if not frameOK:
            continue

        # Display the resulting frame.
        print('FPS %d'%fps, flush=True)
        cv2.imshow('Video [press q to quit]', frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exiting...', flush=True)
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
