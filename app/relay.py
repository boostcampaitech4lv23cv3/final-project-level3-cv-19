from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse


app = FastAPI()

FORWARD_DATA = bytes()  # client로부터 수신한 MJPEG 데이터
boundary = str()  # 이미지 구분용 바운더리
receiving = False  # 데이터 수신 상태


# client에서 보낸 영상을 비동기로 전달
def get_camera_stream():
    global FORWARD_DATA, receiving
    while receiving:
        yield FORWARD_DATA


# client -> API 서버로 스트리밍 전송
@app.post("/relay")
async def relay(request: Request):
    global FORWARD_DATA, boundary, receiving
    boundary = request.headers.get('boundary')  # client 에서 수신한 바운더리 값을 글로벌 변수로 저장

    # MJPEG 영상 수신
    receiving = True
    async for chunk in request.stream():
        FORWARD_DATA = chunk
    receiving = False

    # 영상 수신 종료
    FORWARD_DATA = None
    return {"result": "success"}


# 웹 View
@app.get("/relay")
def web_view():  # client 에서 수신한 boundary 정보와 MJPEG 영상을 스트리밍으로 송신
    global boundary
    return StreamingResponse(get_camera_stream(), media_type=f'multipart/x-mixed-replace; boundary={boundary}')
"""

# Socket Comm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

app = FastAPI()


@app.websocket("/relay")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Connection listen
    try:
        while True:
            contents = await websocket.receive_bytes()
            arr = np.frombuffer(contents, np.uint8)
            frame = cv2.imencode(arr, cv2.IMREAD_COLOR)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
    except WebSocketDisconnect:
        cv2.destroyWindow("frame")
        print("Client Disconnected")
"""
