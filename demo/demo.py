# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
import pandas as pd

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
# from vision.fair.detectron2.demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    print('args.confidence_threshold ', args.confidence_threshold)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def xyxy_to_xywh(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    model = DefaultPredictor(cfg)

    pred_json = []
    pred_csv = []


    if len(args.input) == 1:
        
        args.input = glob.glob(f'{args.input}/*.png') #/mnt/SSD7/yuwei-hdd3/selected/HW2/nycu-hw2-data/test
        args.input = glob.glob('/mnt/SSD7/yuwei-hdd3/selected/HW2/nycu-hw2-data/test/*.png')
        # print(args.input)
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        image_id = int(os.path.basename(path).replace('.png',''))
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions = model(img)

        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()

        digits = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                x_min, y_min, x_max, y_max = box
                box_xywh = xyxy_to_xywh([x_min, y_min, x_max, y_max])
                pred_json.append({
                    "image_id": image_id,
                    "bbox": [round(float(x), 2) for x in box_xywh],
                    "score": round(float(score), 3),
                    "category_id": int(label + 1)  # Detectron2 categories are 0-indexed
                })
                digits.append((x_min, y_min, int(label)))

        if digits:
            digits = sorted(digits, key=lambda x: (x[0], x[1]))  # x_min then y_min
            digit_str = ''.join(str(d[2]) for d in digits)
            pred_csv.append({"image_id": image_id, "pred_label": int(digit_str)})
        else:
            pred_csv.append({"image_id": image_id, "pred_label": -1})


    os.makedirs(f'{args.output}', exist_ok=True)
    with open(f"{args.output}/pred.json", "w") as f:
        json.dump(pred_json, f)

    pd.DataFrame(pred_csv).to_csv(f"{args.output}/pred.csv", index=False)


if __name__ == "__main__":
    main()  # pragma: no cover
