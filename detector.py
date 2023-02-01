import io
import os
import cv2
import torch

import sys
import json
import time
import math

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
def detect(src: str,dst: str):
    #f=io.BytesIO(image_bytes)
    #cv2.imdecode(image_bytes,'mp4v')
    #probe = ffmpeg.probe(image_bytes)
    #video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    #width = int(video_stream['width'])
    #height = int(video_stream['height'])
    #cap = cv2.VideoCapture(image_bytes)

    save_batch="save/batch"
    save_result="save/result/img"

    start=time.time()

    #src = _transform_image(image_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoBackend(weights='yolov8n_custom.pt',device=device,dnn=False,fp16=False)
    model.eval()
    stride, names, pt= model.stride, model.names, model.pt
    imgsz=(640,640)
    imgsz=check_imgsz(imgsz,stride=stride)

    dataset=LoadImages(src,imgsz=imgsz,stride=stride,auto=pt,vid_stride=1)
    
    #windows=[]
    #vid_writer=cv2.VideoWriter(result_path,cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    #vid_writer=[None]*100000
    frames=[]
    data={}

    #th_x=[False]*10
    #th_y=[False]*10

    T1 = 0.7 #threshold1 y
    T2_x1 = 0.3 #threshold2 xleft
    T2_x2 =0.7 #threshold2 xright
    T2_y = 0.9 #threshold2 y

    #Td1, Td2 = 0.1, 0.2
    #Ta1, Ta2 = 0.1, 0.2
    

    time_record1=time.time()

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

        preds = non_max_suppression(preds,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic=False,max_det=1000)
        #pred_txt=open('pred.txt','a')
        #pred_txt.write(f'frame id : {frame_idx}\n')

        results=[]

        """im0=im0s.copy()
        p=Path(path)
        txt_file_name, save_path= p.stem, str('save/'+p.name)
        shape=im0s[i].shape if isinstance(im0s,list) else im0s.shape
        pred[:,:4]=scale_boxes(im.shape[2:], preds[:, :4], shape).round()
        result.append(Results(boxes=pred,orig_shape=shape[:2]))

        annotator=Annotator(im0,line_width=2,example=str(names))

        det=results[i].boxes
        box_info=open(f'{txt_file_name}.txt','a')
        box_info.write(f'{frame_idx}\n')

        for d in reversed(det):
            cls, conf = d.cls.squeeze(),d.cond.squeeze()
            c=int(cls)
            label=f'{model.names[c]}{conf:.2f}'
            annotator.box_label(d.xyxy.squeeze(),label,color=colors(c,True))
            box_info.write(f'{d.xyxy.squeeze().tolist()} {model.names[c]}\n')"""

        for i, pred in enumerate(preds):
            p, im0, _ = path,im0s.copy(),getattr(dataset,'frame',0)
            p = Path(p)
            #txt_file_name=p.stem
            json_file_path=f'{p.stem}.json'
            save_path=str('save/'+p.name)
            #pred_txt.write(f'preds : {pred}\n{pred.shape} {len(pred)}\n')
            
            shape = im0s[i].shape if isinstance(im0s, list) else im0s.shape
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
            
            #pred_txt.write(f'pred : {pred}\n')
            #pred_txt.write(f'results : {Results(boxes=pred, orig_shape=shape[:2])}\n')
            #pred_txt.write(f'pred 0 :{pred[0,:]}\n')
            #pred_txt.write(f'original : {im.shape}, target: {shape}\n')


            annotator=Annotator(im0,line_width=2,example=str(names))

            det=results[i].boxes
            #box_info=open(f'{txt_file_name}.txt','a')
            #box_json=open(f'{json_file_name}.json','a')
            
            #box_info.write(f'{frame_idx}\n')
            i=0
            
            data[f'{frame_idx}']={}
            with open(f'dist_degree_{p.stem}.txt','a') as mathfile:
                mathfile.write(f'{frame_idx}:\n')
            for d in reversed(det):
                box=d.xyxy.squeeze().tolist()
                if box[3]>shape[0]*T1:
                    warn=2
                    #if box[3]>shape[0]*T2_y and not(box[0]>shape[1]*T2_x2 or box[2]<shape[1]*T2_x1):
                    #    warn=1

                    d_x, d_y = (box[0] + box[2]) / 2 - shape[1] / 2, box[3] - shape[0]
                    dist = math.sqrt(pow(d_x,2)+pow(d_y,2))
                    angle = 90 - math.atan2(-d_y,d_x)*180/math.pi
                    with open(f'dist_degree_{p.stem}.txt','a') as mathfile:
                        mathfile.write(f'{i} {dist:.1f} {angle:.1f} {d.xyxy.squeeze().tolist()}\n')
                    if dist<shape[0]*0.1 or (shape[0]*0.1<dist<shape[0]*0.2 and -45<=angle<=45) or -15<=angle<=15:
                        warn=1


                
                    cls, conf = d.cls.squeeze(), d.conf.squeeze()
                    c=int(cls)
                    label = f'{model.names[c]}{conf:.2f}'
                    #pred_txt.write(f'{d.xyxy.squeeze().tolist()}       {d.xyxy.squeeze().tolist()[3]}\n')
                    annotator.box_label(d.xyxy.squeeze(),label,color=colors(4*(warn-1),True))
                    data[f'{frame_idx}'][f'{i}']={"class":f'{model.names[c]}',"warning_lv":f'{warn}',"location":f'{int(((box[0]+box[2])//2)//(shape[1]//3))}'}
                    
                    #data[f'{frame_idx}']{"frame_idx":{i:{"class":cls,"warning_lv":warn,"location":(box[0]+box[2])%(shape[1]//3)}}}
                    i=i+1
            
                
                #annotator.box_label(d.xyxy.squeeze(),label,color=colors(c,True))
                #box_info.write(f'{d.xyxy.squeeze().tolist()} {model.names[c]}\n')

        im0 = annotator.result()
        frames.append(im0)
        
        time_record2=time.time()
        
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
    
    with open(json_file_path,'w') as file:
        json.dump(data,file,indent=4)
    json_obj=json.dumps(data,ensure_ascii=False,indent=None,sort_keys=True)

    vid_writer=cv2.VideoWriter(str(p.name),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    
    for i in range(len(frames)):
        #print(i)
        #cv2.imwrite(f"save/result/{p.name}/img-{i:04}.jpg",frames[i])
        vid_writer.write(frames[i])

    vid_writer.release()

    h264_encoding(str(p.name),dst)
    # return open(result_path, 'rb')

    with open(f'{p.stem}.txt','w') as file:
        file.write(f'model set: {time_record1-start:.4f} sec, inference:{time_record2-time_record1:.4f} sec, write & encoding:{time.time()-time_record2:.4f} sec\n')
    return dst,json_obj

#detect()