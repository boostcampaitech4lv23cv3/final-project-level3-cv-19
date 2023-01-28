import io
import os
import cv2
import torch

import sys

import numpy as np
from pathlib import Path
from yolov8.ultralytics.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.ultralytics.yolo.data.dataloaders.stream_loaders import LoadStreams,LoadImages
#from yolov8.ultralytics.yolo.data.utils import VID_FORMATS
from yolov8.ultralytics.ultralytics.yolo.utils.checks import check_imgsz
from yolov8.ultralytics.ultralytics.yolo.utils.plotting import Annotator, colors
from yolov8.ultralytics.ultralytics.yolo.utils.ops import scale_boxes,non_max_suppression
from yolov8.ultralytics.ultralytics.yolo.engine.results import Results
from yolov8.ultralytics.ultralytics.yolo.utils.files import increment_path

from ffmpeg_func import h264_encoding

import ffmpeg

#def detect(src='./final-project-level3-cv-19-feature-videoclip/app/uploaded/test.mp4'):
#def detect(src: str, result_path: str="test.mp4"):
def detect(src: str):
    #f=io.BytesIO(image_bytes)
    #cv2.imdecode(image_bytes,'mp4v')
    #probe = ffmpeg.probe(image_bytes)
    #video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    #width = int(video_stream['width'])
    #height = int(video_stream['height'])
    #cap = cv2.VideoCapture(image_bytes)

    save_batch="save/batch"
    save_result="save/result/img"

    #src = _transform_image(image_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoBackend(weights='./yolov8/yolov8n_custom.pt',device=device,dnn=False,fp16=False)
    model.eval()
    stride, names, pt= model.stride, model.names, model.pt
    imgsz=(640,640)
    imgsz=check_imgsz(imgsz,stride=stride)

    dataset=LoadImages(src,imgsz=imgsz,stride=stride,auto=pt,vid_stride=1)
    
    #windows=[]
    #vid_writer=cv2.VideoWriter(result_path,cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    #vid_writer=[None]*100000
    frames=[]

    for frame_idx,batch in enumerate(dataset):
        path,im,im0s,vid_cap,s=batch
        ###
        #cv2.imwrite(str(increment_path(save_batch+"_im/img")).with_suffix('.jpg'),im)
        #cv2.imwrite(str(increment_path(save_batch+"_im0s/img")).with_suffix('.jpg'),im0s)
        ###
        im=torch.from_numpy(im)
        
        im=im.float()
        im/=255.0
        if len(im.shape)==3:
            im=im[None]
        
        im=im.to(device)
    
        preds = model(im)

        p = non_max_suppression(preds,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic=False,max_det=1000)

        results=[]

        for i, pred in enumerate(p):
            p, im0, _ = path,im0s.copy(),getattr(dataset,'frame',0)
            p = Path(p)
            txt_file_name=p.stem
            save_path=str('save/'+p.name)
            
            shape = im0s[i].shape if isinstance(im0s, list) else im0s.shape
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))

            annotator=Annotator(im0,line_width=2,example=str(names))

            det=results[i].boxes

            for d in reversed(det):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c=int(cls)
                label = f'{model.names[c]}{conf:.2f}'
                annotator.box_label(d.xyxy.squeeze(),label,color=colors(c,True))

        im0 = annotator.result()
        frames.append(im0)
        
        ###
        #cv2.imwrite(str(increment_path(save_result)).with_suffix('.jpg'),im0)
        ###

        """
        #windows.append(preds)
        if (frame_idx%30) == 0:
            cv2.namedWindow(str(preds),cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(str(p),im0.shape[1],im0.shape[0])
            cv2.imshow(str(p),im0)
            if cv2.waitKey(1) == ord('q'):
                exit()

        if isinstance(vid_writer[frame_idx], cv2.VideoWriter):
            vid_writer[frame_idx].release()
        """
        fps=int(vid_cap.get(cv2.CAP_PROP_FPS))
        w=int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        #save_path=str(Path(save_path).with_suffix('.mp4'))
        #print(save_path,fps,w,h)
        #vid_writer[frame_idx]=cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        #vid_writer[frame_idx].write(im0)
        #print(vid_writer[frame_idx])
        #vid_writer.write(im0)
    
    vid_writer=cv2.VideoWriter(str(p.name),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    
    for i in range(len(frames)):
        #print(i)
        #cv2.imwrite(f"save/result/{p.name}/img-{i:04}.jpg",frames[i])
        vid_writer.write(frames[i])

    vid_writer.release()

    # return open(result_path, 'rb')
    return str(p.name)

#detect()
