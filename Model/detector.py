import os
import cv2
import torch
import json
import math
from pathlib import Path
from app.utils import dir_func

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils.ops import scale_boxes, non_max_suppression
from ultralytics.yolo.engine.results import Results

PRJ_ROOT_PATH = Path(__file__).parent.parent.absolute()
MODEL_DIR = os.path.join(PRJ_ROOT_PATH, "Model")
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")
SAVE_PATH = os.path.join(MODEL_DIR, "save")


def find_location_idx(img_w, x_min, x_max):
    left_th = img_w // 3
    center_th = img_w * 2 // 3
    x_center = (x_min + x_max) // 2
    if x_center < left_th:
        return 0
    elif x_center < center_th:
        return 1
    else:
        return 2


def distance_heading(img_w, img_h, x_min, x_max, y_min, y_max):
    delta_x = (x_min + x_max) / 2 - img_w / 2
    delta_y = y_max - img_h
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    heading = -math.atan2(-delta_y, delta_x) * 180 / math.pi
    return distance, heading


def detect(src: str, session_id: str, conf_thres=0.25, THRESHOLD_y=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Inference with', device)

    tmp_path = os.path.join(APP_PATH, "tmp", session_id)
    img_dst = os.path.join(tmp_path, "img_dir")
    TXT_FILE = os.path.join(tmp_path, f'dist_degree.txt')
    dir_func(img_dst, rmtree=False, mkdir=True)

    model = AutoBackend(weights=os.path.join(MODEL_DIR, "yolov8", 'yolov8n_custom.pt'), device=device, dnn=False, fp16=False)
    model.eval()

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz((640, 640), stride=stride)
    dataset = LoadImages(src, imgsz=imgsz, stride=stride, auto=pt, vid_stride=1)

    json_obj = {}
    dist_T1 = 0.1
    dist_T2 = 0.2
    angle_T1 = 45
    angle_T2 = 15

    for frame_idx, batch in enumerate(dataset, 1):
        _, img_nparr, im0s, vid_cap, s = batch
        annotator = Annotator(im0s, line_width=2, example=str(names))
        img_nparr = torch.from_numpy(img_nparr)

        img_nparr = img_nparr.float()
        img_nparr /= 255.0
        if len(img_nparr.shape) == 3:
            img_nparr = img_nparr[None]

        img_nparr = img_nparr.to(device)
        bboxes_data = non_max_suppression(model(img_nparr), conf_thres=conf_thres, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        for i, bbox in enumerate(bboxes_data):
            shape = im0s[i].shape if isinstance(im0s, list) else im0s.shape
            img_h, img_w, _ = shape
            bbox[:, :4] = scale_boxes(img_nparr.shape[2:], bbox[:, :4], (img_h, img_w)).round()

            json_obj[f'{frame_idx:04d}'] = {}
            # warn_obj=[]
            with open(TXT_FILE, 'a') as f:
                f.write(f'{frame_idx:04d}:\n')
                for obj_id, obj in enumerate(reversed(Results(boxes=bbox, orig_shape=(img_h, img_w)).boxes), 1):
                    bbox = obj.xyxy.squeeze()
                    x_min, y_min, x_max, y_max = bbox_list = bbox.tolist()
                    if y_max > img_h * THRESHOLD_y:
                        dist, angle = distance_heading(img_w, img_h, *bbox_list)

                        f.write(f'{obj_id:02d} {dist:.1f} {angle:.1f} {bbox}\n')
                        # if dist < img_h * 0.1 or (img_h * 0.1 < dist < img_h * 0.2 and -45 <= angle <= 45) or -15 <= angle <= 15:
                        #    warn = 1
                        if dist <= img_h * dist_T1 or (
                                img_h * dist_T1 < dist <= img_h * dist_T2 and -angle_T1 <= angle <= angle_T1) or -angle_T2 <= angle <= angle_T2:
                            warn = 1
                            # warn_obj.append((c, warn, int((((x_min + x_max)/ 2 - (img_w / 2)) /(img_h * 0.1)+3 )// 2), dist, angle))
                        else:
                            warn = 2
                        cls, conf = obj.cls.squeeze(), obj.conf.squeeze()
                        c = int(cls)
                        label = f'{model.names[c]}'
                        annotator.box_label(bbox, label, color=colors(4 * (warn - 1), True))

                        json_obj[f'{frame_idx:04d}'][f'{obj_id:02d}'] = {"class": f'{model.names[c]}',
                                                                         "warning_lv": f"{warn}",
                                                                         "location": f'{find_location_idx(img_w, x_min, x_max)}',
                                                                         "distance": round(dist, 2),
                                                                         "heading": round(angle, 1)}

        cv2.imwrite(os.path.join(img_dst, f"{frame_idx:04}.jpg"), annotator.result())

    return json.dumps(json_obj, ensure_ascii=False, indent=None, sort_keys=True)
