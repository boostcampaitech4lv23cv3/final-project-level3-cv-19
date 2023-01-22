import os
import streamlit as st
import requests
from PIL import Image
from pathlib import Path

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "assets")
API_SERVER_ADDR = ""

st.set_page_config(layout="centered")


def main():
    st.title("시각장애인 보행 시 장애물 안내 서비스")
    req = requests.post(API_SERVER_ADDR, )