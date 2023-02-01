"""
This Codes Temporarily Used for Develop Convenient Purpose.
Not Used from 01Feb2023 because of Server Integration.
"""
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

application = FastAPI()
req_url = "http://127.0.0.1:8001"  # API_SERVER_ADDRESS
# ==================================== 중계 시 필요한 변수 및 함수들
forward_data = bytes()
boundary = str()
receiving = False


def get_stream():
    global forward_data, receiving
    while receiving:
        yield forward_data


# ==================================== 원본 동영상을 중계 서버로 송신
@application.post("/relay")
async def relayed(req: Request = requests.get(req_url + "/original", stream=True)):
    global forward_data, boundary, receiving
    boundary = req.headers.get("boundary")
    receiving = True
    async for chunk in req.stream():
        forward_data = chunk
    receiving = False
    forward_data = None

    return {"result": "success"}


# ==================================== 중계 영상 확인용
@application.get("/relay")
def relaying():
    global boundary
    return StreamingResponse(get_stream(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")
