import requests
import app.cv2_mod

from datetime import datetime
from uuid import UUID, uuid4
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class Item(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str


class Items(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    results: List[Item] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


def video_stream():
    return app.cv2_mod.get_stream_video()


application = FastAPI()
# req_url = "{GPU_SERVER_ADDRESS}"
req_url = "127.0.0.1:8002/relay"


@application.post("/inference")
async def request_inference():
    req = requests.post(req_url, data=StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame"))
    return req
