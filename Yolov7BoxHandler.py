import argparse

import torch
from models.experimental import attempt_load

from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.torch_utils import select_device


class Yolov7BoxHandler:
    """
    This class handles all logic for Yolov7 model for people detection
    """
    def __init__(self, weights="yolov7.pt", imgsz=640, conf_thres=0.25, iou_thres=0.45, agnostic_nms=True, augment=True,
                 device=""):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.device = device

    def detect(self, source: str) -> list:
        """
        Detects people in images located in source folder
        :param source: folder with images
        :return: list of lists of boxes for each image
        """
        with torch.no_grad():
            # Initialize
            device = select_device(self.device)

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

            dataset = LoadImages(source, img_size=imgsz, stride=stride)

            answer_persons = []

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0,
                                           agnostic=self.agnostic_nms)

                # Process detections
                tmp_answer_persons = []
                for i, det in enumerate(pred):  # detections per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if cls == 0:
                                tmp_answer_persons.append(xyxy)
                answer_persons.append(tmp_answer_persons)

            return answer_persons
