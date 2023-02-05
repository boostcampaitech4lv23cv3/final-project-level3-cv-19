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
        
        cap = cv2.VideoCapture(src)
        framecount = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        json_obj = {}
        print(f'TensorRT inference source:{src}') 
        print(f'framecount:{framecount}, fps:{fps}, width:{img_w}, height:{img_h}, confidence threshold:{conf_thres}')

        mask_h,mask_w = img_h, img_w
        mask = np.zeros((mask_h,mask_w, 3),np.uint8)
        mask_thres1 = np.zeros((mask_h,mask_w),np.uint8)
        mask_thres2 = np.zeros((mask_h,mask_w),np.uint8)

        #threshold line settings
        far = 0.18
        middle = 0.145
        near = 0.1

        #threshold angle settings
        angle_far = 15
        angle_middle = 30
        angle_near = 90
        angle_center = 270

        #ellipse ratio
        ellipse_value = 2.5

        #color settings
        YELLOW = (0,255,255)
        RED = (0,0,255)

        #masking value
        bitmask = 255

        #mask = np.zeros((720,1280,3),np.uint8)
        cv2.ellipse(mask,(int(mask_w/2),mask_h),(int(mask_h * far * ellipse_value),int(mask_h * far)),0,angle_center - angle_near,angle_center + angle_near,YELLOW,-1)
        cv2.ellipse(mask,(int(mask_w/2),mask_h),(int(mask_h * near * ellipse_value),int(mask_h * near)),0,angle_center - angle_near,angle_center + angle_near,RED,-1)
        cv2.ellipse(mask,(int(mask_w/2),mask_h),(int(mask_h * middle * ellipse_value),int(mask_h * middle)),0,angle_center - angle_middle,angle_center + angle_middle,RED,-1)
        cv2.ellipse(mask,(int(mask_w/2),mask_h),(int(mask_h * far * ellipse_value),int(mask_h * far)),0,angle_center - angle_far,angle_center + angle_far,RED,-1)

        #mask_thres1
        cv2.ellipse(mask_thres1,(int(mask_w/2),mask_h),(int(mask_h * far * ellipse_value),int(mask_h * far)),0,angle_center - angle_near,angle_center + angle_near,bitmask,-1)
        #mask_thres2
        cv2.ellipse(mask_thres2,(int(mask_w/2),mask_h),(int(mask_h * near * ellipse_value),int(mask_h * near)),0,angle_center - angle_near,angle_center + angle_near,bitmask,-1)
        cv2.ellipse(mask_thres2,(int(mask_w/2),mask_h),(int(mask_h * middle * ellipse_value),int(mask_h * middle)),0,angle_center - angle_middle,angle_center + angle_middle,bitmask,-1)
        cv2.ellipse(mask_thres2,(int(mask_w/2),mask_h),(int(mask_h * far * ellipse_value),int(mask_h * far)),0,angle_center - angle_far,angle_center + angle_far,bitmask,-1)

        import time
        t1 = time.time()
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = self.preproc(frame, self.imgsz, self.mean, self.std)
            
            data = self.infer(blob) # run inference by tensorRT

            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), 
                                                            np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if ((frame_idx % (4*fps)) == 0):
                print(f'Inference is in progress {frame_idx}/{framecount}')

            if dets is not None and dets.size > 0:
                json_obj[f'{frame_idx:04d}'] = {}
                with open(TXT_FILE, 'a') as f:
                    f.write(f'{frame_idx:04d}:\n')

                    #timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
                    final_warns = [3] * dets.size
                    for obj_id, [x_min, y_min, x_max, y_max, conf, cls] in enumerate(dets):
                        #print(f'{src}: Frame:{frame_idx} ObjIndex:{obj_id} Time:{timestamp:.2f}--Class:{int(cls)} Warning:{int(warn)}')
                        x_min, y_min, x_max, y_max = self.fit2img(img_w, img_h, x_min, y_min, x_max, y_max)
                        if y_max > img_h * THRESHOLD_y:
                            bbox_list = [x_min, y_min, x_max, y_max]
                            dist, angle = self.distance_heading(img_w, img_h, *bbox_list)

                            warn=3

                            np_size = (int(y_max-y_min),int(x_max-x_min))
                            if np.any((mask_thres1[int(y_min):int(y_max),int(x_min):int(x_max)] & np.ones(np_size,np.uint8)) > 0):
                                warn = 2
                                if np.any((mask_thres2[int(y_min):int(y_max),int(x_min):int(x_max)] & np.ones(np_size,np.uint8)) > 0):
                                    warn = 1
                            
                            json_obj[f'{frame_idx:04d}'][f'{obj_id:02d}'] = {"class": f'{self.class_names[int(cls)]}',
                                                                                "warning_lv": f"{warn}",
                                                                                "location": f'{self.find_location_idx(img_w, x_min, x_max)}',
                                                                                "distance": round(dist, 2),
                                                                                "heading": round(angle, 1)}
                            final_warns[obj_id] = warn                                                                
                    frame = self.vis(frame, final_boxes, final_scores, final_cls_inds, final_warns,    # disable when time measurement
                                    conf=conf_thres, class_names=self.class_names)                     # disable when time measurement                                       
            cv2.imwrite(os.path.join(img_dst, f"{frame_idx:04}.jpg"), cv2.addWeighted(mask, 0.2, frame,0.8,0))# disable when time measurement
            #cv2.imwrite(os.path.join(img_dst, f"{frame_idx:04}.jpg"), frame)# enable when time measurement
        cap.release()
        
        elapsedtime = time.time() - t1
        print(f'Inference time {elapsedtime}/Frames {framecount}') # time measurement

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
            #text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)  # class name:score%
            text = class_names[cls_id] # class name only
            _COLORS = self.rainbow_fill(80).astype(np.float32).reshape(-1, 3)
            #txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            txt_color = (255, 255, 255)
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
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.65, txt_color, thickness=1)

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


    def fit2img(self, img_w, img_h, x_min, y_min, x_max, y_max):
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        if(x_min < 0):
            x_min = 0
        if(y_min < 0):
            y_min = 0
        if(x_max >= img_w):
            x_max = img_w - 1
        if(y_max >= img_h):
            y_max = img_h - 1

        return x_min, y_min, x_max, y_max


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


