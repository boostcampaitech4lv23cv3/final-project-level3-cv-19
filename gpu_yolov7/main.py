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

app = FastAPI()

@app.post("/results")
async def predict(request: Request, files: List[UploadFile] = File(...)):
    for file in files:
        # head = dict(request.headers)
        # file_name = head["filename"]
        file_name = file.filename
        # user_id = head["user_id"]
        video_bytes = await file.read()
        result = None
        with torch.no_grad():
            # detect(source='gpu_dummy/input/test.mp4')
            with open(file_name, 'wb') as f:
                f.write(video_bytes)
            result = detect(source=file_name)
        return Response(content=result, media_type="video/mp4", headers={"filename": file_name})