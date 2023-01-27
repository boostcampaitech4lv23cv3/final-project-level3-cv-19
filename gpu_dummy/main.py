from typing import List
from fastapi import FastAPI, UploadFile, File, Response

app = FastAPI()


@app.post("/results")
async def predict(files: List[UploadFile] = File(...)):
    for file in files:
        video_bytes = await file.read()
        return Response(content=video_bytes, media_type="video/mp4")
