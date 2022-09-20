import os

from KeyPointsHandler import get_homography_matrix
from Yolov7BoxHandler import Yolov7BoxHandler
import cv2

from draw import drawBoxes


def main():
    # vcap = cv2.VideoCapture('sample.mp4')
    # os.makedirs('frames', exist_ok=True)
    # i = 0
    # while True:
    #     i += 1
    #     if i > 3:
    #         exit(0)
    #     ret, frame = vcap.read()
    #     if frame is not None:
    #         cv2.imwrite('frames/frame' + str(i) + '.jpg', frame)
    #         if cv2.waitKey(22) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    yolo = Yolov7BoxHandler()
    boxes = yolo.detect("frames")
    print(get_homography_matrix("frames/frame1.jpg", "frames/frame2.jpg", boxes[0], boxes[1]))
    # print(boxes[0])
    # drawBoxes(boxes[0], "frames/frame1.jpg")

main()
