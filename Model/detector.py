import os
import cv2
import torch
import json
import time
import math
from pathlib import Path
from app.ffmpeg_func import h264_encoding

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


def detect(src: str, session_id:str, dst: str):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Inference by', device)
    model = AutoBackend(weights=os.path.join(MODEL_DIR, "yolov8", 'yolov8n_custom.pt'), device=device, dnn=False, fp16=False)

    model.eval()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (640, 640)
    imgsz = check_imgsz(imgsz, stride=stride)

    dataset = LoadImages(src, imgsz=imgsz, stride=stride, auto=pt, vid_stride=1)
    frames = []
    data = {}

    T1 = 0.7  # threshold1 y
    dist_T1=0.1
    dist_T2=0.2
    angle_T1=45
    angle_T2=15

    time_record1 = time.time()
    # Local Variables Assignment
    tmp_path = os.path.join(APP_PATH, "tmp", session_id)
    json_file_path = None
    p = None
    fps = None
    w, h = None, None
    time_record2 = None

    for frame_idx, batch in enumerate(dataset, 1):
        path, im, im0s, vid_cap, s = batch
        annotator = None
        im = torch.from_numpy(im)

        im = im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        im = im.to(device)
        preds = model(im)
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        results = []

        for i, pred in enumerate(preds):
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            json_file_path = os.path.join(tmp_path, f'{p.stem}.json')

            shape = im0s[i].shape if isinstance(im0s, list) else im0s.shape
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
            annotator = Annotator(im0, line_width=2, example=str(names))
            det = results[i].boxes
            j = 0
            data[f'{frame_idx:04d}'] = {}
            warn_obj=[]
            TXT_FILE = os.path.join(tmp_path, f'dist_degree_{p.stem}.txt')

            with open(TXT_FILE, 'a') as mathfile:
                mathfile.write(f'{frame_idx:04d}:\n')
            for d in reversed(det):
                box = d.xyxy.squeeze().tolist()
                if box[3] > shape[0] * T1:
                    cls, conf = d.cls.squeeze(), d.conf.squeeze()
                    c = int(cls)
                    label = f'{model.names[c]}{conf:.2f}'
                    
                    warn = 2
                    d_x, d_y = (box[0] + box[2]) / 2 - shape[1] / 2, box[3] - shape[0]
                    dist = math.sqrt(pow(d_x, 2) + pow(d_y, 2))
                    angle = 90 - math.atan2(-d_y, d_x) * 180 / math.pi
                    with open(TXT_FILE, 'a') as mathfile:
                        mathfile.write(f'{j} {dist:.1f} {angle:.1f} {d.xyxy.squeeze().tolist()}\n')
                    if dist <= shape[0] * dist_T1 or (
                            shape[0] * dist_T1 < dist <= shape[0] * dist_T2 and -angle_T1 <= angle <= angle_T1) or -angle_T2 <= angle <= angle_T2:
                        warn = 1
                        warn_obj.append((c,warn,int((((box[0] + box[2])/ 2 - (shape[1] / 2)) /(shape[0] * 0.1)+3 )// 2),dist))
                        

                    annotator.box_label(d.xyxy.squeeze(), label, color=colors(4 * (warn - 1), True))
                    j += 1
            
            if len(warn_obj):
                warn_obj.sort(key=lambda x:x[3])
                data[f'{frame_idx:04d}'][0]={"class": f'{model.names[warn_obj[0][0]]}', "warning_lv": f'{warn_obj[0][1]}',
                                                        "location": f'{warn_obj[0][2]}'}

        im0 = annotator.result()
        frames.append(im0)
        time_record2 = time.time()

        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    json_obj = json.dumps(data, ensure_ascii=False, indent=None, sort_keys=True)

    vid_writer = cv2.VideoWriter(str(p.name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(len(frames)):
        vid_writer.write(frames[i])
    vid_writer.release()

    h264_encoding(str(p.name), dst)

    with open(os.path.join(tmp_path, f'{p.stem}.txt'), 'w') as file:
        file.write(
            f'model set: {time_record1 - start:.4f} sec, inference:{time_record2 - time_record1:.4f} sec, write & encoding:{time.time() - time_record2:.4f} sec\n')
    return dst, json_obj
