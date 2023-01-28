from typing import List
from fastapi import FastAPI, UploadFile, File, Request, Response

app = FastAPI()


@app.post("/")
async def predict(request: Request, files: List[UploadFile] = File(...)):
    head = request.headers
    file_name = head.get("filename")
    user_id = head.get("user_id")
    for file in files:
        video_bytes = await file.read()

        return Response(content=video_bytes, media_type="video/mp4", headers={"filename": file_name, "user_id": user_id})
