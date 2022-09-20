import cv2
import numpy as np

#
# def get_keypoints(path):
#     img = cv2.imread(path)
#     sift = cv2.SIFT()
#     kp1, des1 = sift.detectAndCompute(img, None)
#     return kp1, des1
#
#
# def get_matches(path1, path2):
#     kp1, des1 = get_keypoints(path1)
#     kp2, des2 = get_keypoints(path2)
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append([m])
#     return good, kp1, kp2
#
#
# def get_homography_matrix(path1, path2, boxes1, boxes2):
#     good, kp1, kp2 = get_matches(path1, path2)
#     if len(good) > 4:
#         src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         return M
#     else:
#         return None
from draw import drawBoxes


def get_homography_matrix(path1, path2, boxes1, boxes2):
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(30000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2, None))

    # Draw matches
    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches_example/matches.jpg", im_matches)

    # Extract location of matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Filter points by boxes
    good_matches = []

    for i, match in enumerate(matches):
        flag = False
        for box1 in boxes1:
            for box2 in boxes2:
                if ((int(box1[0]) < int(points1[i][0]) < int(box1[2])) and (
                        int(box1[1]) < int(points1[i][1]) < int(box1[3]))) \
                        or ((int(box2[0]) < int(points2[i][0]) < int(box2[2])) and (
                        int(box2[1]) < int(points2[i][1]) < int(box2[3]))):
                    flag = True
                    break

            if flag:
                break
        if not flag:
            good_matches.append(match)

    # Sort matches by score
    good_matches.sort(key=lambda x: x.distance, reverse=False)
    good_matches = good_matches[:100]

    # Extract location of good matches
    good_points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    good_points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        good_points1[i, :] = keypoints1[match.queryIdx].pt
        good_points2[i, :] = keypoints2[match.trainIdx].pt

    # Draw top matches
    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, None)
    drawBoxes(boxes1, im_matches)
    cv2.imwrite("matches_example/matches_after_boxes.jpg", im_matches)

    # Find homography
    M, mask = cv2.findHomography(good_points1, good_points2, cv2.RANSAC, 5.0)

    return M
