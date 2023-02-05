import pathlib
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("live_server:app", host="0.0.0.0", port=30002)

origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
RESULT_PATH = os.path.join(PRJ_ROOT_PATH, "app", "result")


@app.get("/{user_id}/video")
async def video(user_id):
    video_path = pathlib.Path(os.path.join(RESULT_PATH, user_id, "result.mp4"))
    return FileResponse(video_path)


@app.get("/{user_id}/subtitle")
async def subtitle(user_id):
    subtitle_path = pathlib.Path(os.path.join(RESULT_PATH, user_id, "result.vtt"))
    return FileResponse(subtitle_path)
