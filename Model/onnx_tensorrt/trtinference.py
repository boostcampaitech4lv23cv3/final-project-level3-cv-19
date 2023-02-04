import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import math
from pathlib import Path
from utils import dir_func # app.
import argparse
from ultralytics.yolo.utils.plotting import Annotator, colors

PRJ_ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
MODEL_DIR = os.path.join(PRJ_ROOT_PATH, "Model")
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")
SAVE_PATH = os.path.join(MODEL_DIR, "save")


class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 29
        self.class_names = ['bicycle','bus','car','carrier','cat','dog','motorcycle','movable_signage','person',
                'scooter','stroller','truck','wheelchair','barricade','bench','bollard','chair','fire_hydrant',
                'kiosk','parking_meter','pole','potted_plant','power_controller','stop','table','traffic_light',
                'traffic_light_controller','traffic_sign','tree_trunk']

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data


    def detect_video(self, src: str, session_id:str, conf_thres=0.25, THRESHOLD_y=0.7) -> str:
        
        tmp_path = os.path.join(APP_PATH, "tmp", session_id)
        img_dst = os.path.join(tmp_path, "img_dir")
        TXT_FILE = os.path.join(tmp_path, f'dist_degree.txt')
        JSON_FILE = os.path.join(tmp_path, 'objdetection.json')
        dir_func(img_dst, rmtree=False, mkdir=True)
        
        cap = cv2.VideoCapture(src)
        framecount = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        json_obj = {}
        inf_fps = 0
        dist_T1 = 0.1
        dist_T2 = 0.2
        angle_T1 = 45
        angle_T2 = 15
        print(f'TensorRT inference source:{src}') 
        print(f'framecount:{framecount}, fps:{fps}, width:{img_w}, height:{img_h}')
        import time
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = self.preproc(frame, self.imgsz, self.mean, self.std)
            
            #t1 = time.time()
            data = self.infer(blob) # run inference by tensorRT
            #inf_fps = (inf_fps + (1. / (time.time() - t1))) / 2
            #frame = cv2.putText(frame, "FPS:%d " %inf_fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                    (0, 0, 255), 2)

            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), 
                                                            np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            dets = self.getwarningdets(dets, img_w, img_h)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if ((frame_idx % (4*fps)) == 0):
                print(f'Inference is in progress {frame_idx}/{framecount}')

            if dets is not None and dets.size > 0:
                json_obj[f'{frame_idx:04d}'] = {}
                with open(TXT_FILE, 'a') as f:
                    f.write(f'{frame_idx:04d}:\n')

                    final_boxes, final_scores, final_cls_inds, final_warn_inds = dets[:,:4], dets[:, 4], dets[:, 5], dets[:,6]
                    frame = self.vis(frame, final_boxes, final_scores, final_cls_inds, final_warn_inds,
                                    conf=conf_thres, class_names=self.class_names)
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                    for obj_id, [x_min, y_min, x_max, y_max,conf,cls,warn] in enumerate(dets):
                        #print(f'{src}: Frame:{frame_idx} ObjIndex:{obj_id} Time:{timestamp:.2f}--Class:{int(cls)} Warning:{int(warn)}')
                        
                        if y_max > img_h * THRESHOLD_y:
                            bbox_list = [x_min, y_min, x_max, y_max]
                            dist, angle = self.distance_heading(img_w, img_h, *bbox_list)

                            f.write(f'{obj_id:02d} {dist:.1f} {angle:.1f} {bbox_list}\n')
                            # if dist < img_h * 0.1 or (img_h * 0.1 < dist < img_h * 0.2 and -45 <= angle <= 45) or -15 <= angle <= 15:
                            #    warn = 1
                            if dist <= img_h * dist_T1 or (
                                    img_h * dist_T1 < dist <= img_h * dist_T2 and -angle_T1 <= angle <= angle_T1) or -angle_T2 <= angle <= angle_T2:
                                warn = 1
                                # warn_obj.append((c, warn, int((((x_min + x_max)/ 2 - (img_w / 2)) /(img_h * 0.1)+3 )// 2), dist, angle))
                            else:
                                warn = 2
                            
                            json_obj[f'{frame_idx:04d}'][f'{obj_id:02d}'] = {"class": f'{self.class_names[int(cls)]}',
                                                                                "warning_lv": f"{warn}",
                                                                                "location": f'{self.find_location_idx(img_w, x_min, x_max)}',
                                                                                "distance": round(dist, 2),
                                                                                "heading": round(angle, 1)}
            cv2.imwrite(os.path.join(img_dst, f"{frame_idx:04}.jpg"), frame)
        cv2.destroyAllWindows()
        cap.release()
        
        with open(JSON_FILE, 'w') as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=None, sort_keys=True)

        return 0


    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')

    def getwarningdets(self, dets,width,height):
        warningdets = []
        T1 = 0.7 #threshold1 y
        T2_x1 = 0.3 #threshold2 xleft
        T2_x2 =0.7 #threshold2 xright
        T2_y = 0.9 #threshold2 y

        for x1,y1,x2,y2,score,cls in dets.tolist():
            box=[x1, y1, x2, y2]

            if box[3]>height*T1: # y - close
                warn=2
                d_x, d_y = (box[0] + box[2]) / 2 - width / 2, box[3] - height
                dist = math.sqrt(pow(d_x,2)+pow(d_y,2))
                angle = 90 - math.atan2(-d_y,d_x)*180/math.pi
                
                if dist<height*0.1 or (height*0.1<dist<height*0.2 and -45<=angle<=45) or -15<=angle<=15:
                #if box[3]>height*T2_y and not(box[0]>width*T2_x2 or box[2]<width*T2_x1): # x - center
                    warn=1

                warningdets.append([x1,y1,x2,y2,score,cls,warn])

        warningdets = np.array(warningdets)

        return warningdets

    def preproc(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # if use yolox set
        # padded_img = padded_img[:, :, ::-1]
        # padded_img /= 255.0
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def rainbow_fill(self, size=50):  # simpler way to generate rainbow color
        cmap = plt.get_cmap('jet')
        color_list = []

        for n in range(size):
            color = cmap(n/size)
            color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

        return np.array(color_list)

    def vis(self, img, boxes, scores, cls_ids, warn_idxs, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            warn_idx = int(warn_idxs[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color=colors(4*(warn_idx-1),True)
            #color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            _COLORS = self.rainbow_fill(80).astype(np.float32).reshape(-1, 3)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.65, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            #txt_bk_color = (_
            # [cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=2)

        return img

    def find_location_idx(self, img_w, x_min, x_max):
        left_th = img_w // 3
        center_th = img_w * 2 // 3
        x_center = (x_min + x_max) // 2
        if x_center < left_th:
            return 0
        elif x_center < center_th:
            return 1
        else:
            return 2


    def distance_heading(self, img_w, img_h, x_min, x_max, y_min, y_max):
        delta_x = (x_min + x_max) / 2 - img_w / 2
        delta_y = y_max - img_h
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        heading = -math.atan2(-delta_y, delta_x) * 180 / math.pi
        return distance, heading


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 29

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-s", "--session_id", help="session id")
    parser.add_argument("-c", "--confidence", help="confidence")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    
    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
        origin_img = pred.inference(img_path, conf=args.confidence)

        cv2.imwrite("%s" %args.output , origin_img)
    if video:
        pred.detect_video(video, session_id=args.session_id, conf_thres=float(args.confidence)) # set 0 use a webcam


