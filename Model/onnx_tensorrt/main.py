from typing import List
from fastapi import FastAPI, UploadFile, File, Request, Response
from onnx_tensorrt.utils import BaseEngine
app = FastAPI()


@app.post("/")
async def predict(request: Request, files: List[UploadFile] = File(...)):
    file_name = request.headers.get('filename')
    user_id = request.headers.get('user_id')
    print(user_id)
    for file in files:
        video_bytes = await file.read()
        with open(file_name, 'wb') as f:
            f.write(video_bytes)

        pred = BaseEngine(engine_path='./onnx_tensorrt/yolov8n_custom.trt')
        result, json_obj = pred.detect_video(file_name=file_name, conf=0.1, end2end=True)
        result_b = None
        with open(result, 'rb') as f:
            result_b = f.read()
        
        return Response(content=result_b, data=json_obj, media_type="video/mp4", 
                                            headers={"filename": file_name, "user_id": user_id})

