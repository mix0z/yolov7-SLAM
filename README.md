# yolov7-SLAM 
# This repository is a fork of yolov7 repository, with some modifications to make SLAM.

## How to use

File `main.py` contains script, that visualizes camera movement for given video
```
python3 main.py <video_name>
```
All videos should be placed in `videos` folder.

Resulting video will be placed in `processed_videos` folder.

While video is being processed, created frames will be placed in `frames` folder.
This was done because computer can run out of memory, if all frames are stored in memory.
So, if you want to process video again, you should delete this video's folder from `frames` folder.

## Description

* In `main.py` function `get_homographs_for_video` is used to get homographs for each frame. It takes video name as an argument and returns list of homographs.
* `Yolov7BoxHndler` class is used to detect person boxes on frames using  YOLOv7.
* File `KeyPointsHandler.py` contains function `get_homography_matrix`, which finds homography matrix for two frames. It takes two frames and two list of person boxes as arguments and returns homography matrix.
To find homography matrix, it uses `cv` methods to find key points, filter them such as no one is in the person box, match and estimate homography matrix.
You may set argument `is_draw` to `True` to see how key points are found, and how they are filtered by boxes. The result will be saved in `matches_example` folder.

## Results

You can see example video in `videos` folder. 
The result of processing this video is in `processed_videos` folder.
And also example of filtered matches in `matches_example` folder.
To reproduce this result, you should run `python3 main.py sample3.mp4` in the root of the repository.

* All matches

![All matches](https://github.com/mix0z/yolov7-SLAM/blob/main/matches_example/matches.jpg)

* Matches filtered by YOLOV7 person boxes

![Matches filtered by YOLOV7 person boxes](https://github.com/mix0z/yolov7-SLAM/blob/main/matches_example/matches_after_boxes.jpg)

* Video example with points which illustrate camera movements

![Video example with points which illustrate camera movements](https://github.com/mix0z/yolov7-SLAM/blob/main/example.png)

## Future work

* Collect dataset with relevant videos
* Optimize parameters for finding and matching key points
* Optimize architecture and algorithms to make it faster
* Add exceptions handling to make it robust

## Contacts

If you have any questions, feel free to contact me via email: `mixoz3101@gmail.com`
or Telegram: `@mix0z`
