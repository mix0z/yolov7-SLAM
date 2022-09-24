import os
from typing import Tuple, List
import sys
import numpy as np

from KeyPointsHandler import get_homography_matrix
from Yolov7BoxHandler import Yolov7BoxHandler
import cv2


def create_points_matrix(w: int, h: int) -> list:
    """
    Creates a matrix of points for visualizing the homography (camera movements)
    :param w: width of the frame
    :param h: height of the frame
    :return: matrix of points
    """
    pointsMatrix = []
    for i in range(-5, 8):
        for j in range(-5, 8):
            pointsMatrix.append([w * i, h * j])
    return pointsMatrix


def split_video_to_frames(video_path: str) -> Tuple[List[str], Tuple[int, int]]:
    """
    Splits the video to frames and saves them in the frames folder
    :return: list of frame names and center of the frame
    """
    vcap = cv2.VideoCapture('videos/' + video_path)
    w, h = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = (int(w / 2), int(h / 2))
    frame_names = []
    dir_name = 'frames/' + video_path.split('.')[0]
    os.makedirs(dir_name, exist_ok=False)
    i = 0
    while True:
        i += 1
        ret, frame = vcap.read()
        if frame is not None:
            cv2.imwrite(dir_name + '/' + str(i) + '.jpg', frame)
            frame_names.append(dir_name + '/' + str(i) + '.jpg')
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
        else:
            break

    return frame_names, center


def get_homographs_for_each_frame(frame_names: list, person_boxes: list) -> list:
    """
    Gets the homography matrix for each frame
    :param frame_names: list of frame names for each frame
    :param person_boxes: list of person boxes for each frame
    :return: list of homography matrices for each frame
    """
    homographs = []
    for i in range(len(frame_names) - 1):
        M = get_homography_matrix(frame_names[i], frame_names[i + 1], person_boxes[i], person_boxes[i + 1])
        homographs.append(M)

    return homographs


def get_points_for_each_frame(frame_names: list, homographs: list, points_matrix: list) -> list:
    """
    Applies the homography matrix to the points matrix and returns the new points for each frame
    :param frame_names: list of frame names
    :param homographs: list of homography matrices
    :param points_matrix: matrix of points
    :return: list of new points for each frame
    """
    points_matrix_arr = [np.array(points_matrix).reshape(-1, 1, 2).astype(np.float32)]
    for i in range(len(frame_names) - 1):
        pts1 = np.array(points_matrix_arr[len(points_matrix_arr) - 1]).reshape(-1, 1, 2).astype(np.float32)
        points_matrix_arr.append(cv2.perspectiveTransform(pts1, homographs[i]))

    return points_matrix_arr


def make_video(video_path: str) -> None:
    """
    Creates a video from the frames
    """
    os.system(
        'ffmpeg -framerate 30 -i frames/' +
        video_path.split('.')[0] +
        '/%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p processed_videos/' +
        video_path.split('.')[0] + '_processed.mp4')


def draw_points_on_frames(frame_names: list, centers: list) -> None:
    """
    Draws the points on the frames
    :param frame_names: list of frame names
    :param centers: list of points for each frame
    """
    for i in range(len(frame_names)):
        img = cv2.imread(frame_names[i])
        for circle in centers[i]:
            cv2.circle(img, (int(circle[0][0]), int(circle[0][1])), 10, (0, 0, 255), -1)
        cv2.imwrite(frame_names[i], img)


def get_homographs_for_video(video_path: str) -> Tuple[List[str], Tuple[int, int], List[np.ndarray]]:
    """
    Gets the homography matrix for each frame
    :return: list of frame names, center of the frame, list of homography matrices for each frame
    """
    frame_names, center = split_video_to_frames(video_path)
    yolo = Yolov7BoxHandler()
    person_boxes = yolo.detect("frames/" + video_path.split('.')[0])

    homographs = get_homographs_for_each_frame(frame_names, person_boxes)
    return frame_names, center, homographs


def main(video_path: str):
    # Get the homography matrix for each frame
    frame_names, center, homographs = get_homographs_for_video(video_path)

    # Visualize camera movements
    points_matrix = create_points_matrix(center[0], center[1])
    centers = get_points_for_each_frame(frame_names, homographs, points_matrix)
    draw_points_on_frames(frame_names, centers)

    # Create a video from the frames
    make_video(video_path)


main(sys.argv[1])
