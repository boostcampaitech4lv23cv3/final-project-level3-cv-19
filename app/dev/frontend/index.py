"""
This Codes Temporarily Used for Develop Convenient Purpose.
Not Used from 01Feb2023 because of Server Integration.
"""
import os
import streamlit as st
import cv2
import websockets
import asyncio
from starlette.responses import StreamingResponse
from starlette.requests import Request
from pathlib import Path

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "assets")
# API_SERVER_ADDR = "localhost:8001/inference"
API_SERVER_ADDR = "http://localhost:8002/relay"
# API_SERVER_ADDR = "ws://localhost:8002/relay"
st.set_page_config(layout="centered")


def main():
    st.title("보행 시 장애물 안내 서비스")
    run = st.checkbox('Run')
    frame_window = st.image([], output_format="JPEG")
    result_window = st.image([], output_format="JPEG")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while run:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
            _, mat_buffer = cv2.imencode(".jpg", frame)
            img_byte = mat_buffer.tobytes()


main()
