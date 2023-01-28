import argparse
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from typing import List
from fastapi import FastAPI, UploadFile, File, Response, Request

import shlex
from subprocess import check_call
import time
from gpu_yolov7.detect import detect
import os
import sys
yolo_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+'/yolov7'
sys.path.append(yolo_path)
from models.experimental import attempt_load
from utils.torch_utils import select_device

app = FastAPI()

#load model
weights = ['gpu_yolov7/best_30e.pt']
device = select_device('')
model = attempt_load(weights, map_location=device)  # load FP32 model

@app.post("/")
async def predict(request: Request, files: List[UploadFile] = File(...)):
    for file in files:
        file_name = file.filename
        user_id = request.headers.get('user_id')
        video_bytes = await file.read()
        result = None
        with open(file_name, 'wb') as f:
            f.write(video_bytes)
        result = detect(source=file_name, model=model)
        return Response(content=result, media_type="video/mp4", headers={"user_id": user_id,"filename": file_name})