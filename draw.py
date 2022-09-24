import random

import cv2
import numpy as np


def drawBoxesByPath(boxes: list, path: str):
    """
    Draws boxes on image by path
    :param boxes: list of boxes
    :param path: path to image
    """
    img = cv2.imread(path)
    drawBoxes(boxes, img)
    # newpath = path.replace("frames", "boxes")
    cv2.imshow("image", img)
    cv2.waitKey(0)
    # cv2.imwrite(newpath, img)


def drawBoxes(boxes: list, img: np.ndarray):
    """
    Draws boxes on image
    :param boxes: list of boxes
    :param img: image
    """
    for x in boxes:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        tl = 3 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = None or [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
