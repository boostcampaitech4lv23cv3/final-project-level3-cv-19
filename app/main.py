import os
import shutil
from app.ffmpeg_func import split_cut, split_segment, concatenate

import requests
import streamlit as st

st.set_page_config(layout="centered")
SERVER_URL = "http://localhost:30002/results"
tmp_path = "app/tmp/"
upload_path = "app/uploaded/"
tmp_rcv_path = "app/tmp_rcv/"
dst_path = "app/result/"


def dir_func(path: str):
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    st.title("보행 시 장애물 안내 서비스")
    uploaded_file = st.file_uploader("동영상을 선택하세요", type=["mp4"])

    if uploaded_file:
        fn = uploaded_file.name
        save_path = os.path.join(upload_path, fn)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            st.success(f"파일이 서버에 저장되었습니다.  {save_path} ")
        st.write("...GPU 서버로 파일 전송 중...")
        idx = 0
        # for file in sorted(split_cut(save_path, 2)):
        for file in sorted(split_segment(save_path, 2)):
            files = [("files", open(file, 'rb'))]
            req = requests.post(SERVER_URL, files=files)
            if req.status_code == 200:
                with open(os.path.join(tmp_rcv_path, f"{idx:04}.mp4"), 'wb') as f:
                    f.write(req.content)
            idx += 1
        rslt_file = os.path.join(dst_path, "result.mp4")
        concatenate(tmp_rcv_path, rslt_file)
        st.video(open(rslt_file, 'rb').read(), format="video/mp4")


dir_func(tmp_path)
dir_func(upload_path)
dir_func(tmp_rcv_path)
dir_func(dst_path)
main()
