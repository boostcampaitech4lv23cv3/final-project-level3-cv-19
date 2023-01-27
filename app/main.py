import os
import shutil
from app.ffmpeg_func import split_cut, split_segment, concatenate

import requests
import streamlit as st
try:
    from streamlit.runtime.runtime import SessionInfo
except ModuleNotFoundError:
    # streamlit < 1.12.1
    try:
        from streamlit.web.server.server import SessionInfo  # type: ignore
    except ModuleNotFoundError:
        # streamlit < 1.12.0
        from streamlit.server.server import SessionInfo  # type: ignore

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ModuleNotFoundError:
    # streamlit < 1.12.0
    try:
        from streamlit.scriptrunner import get_script_run_ctx  # type: ignore
    except ModuleNotFoundError:
        # streamlit < 1.8
        try:
            from streamlit.script_run_context import get_script_run_ctx  # type: ignore
        except ModuleNotFoundError:
            # streamlit < 1.4
            from streamlit.report_thread import (  # type: ignore
                get_report_ctx as get_script_run_ctx,
            )

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


def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")

    return ctx.session_id


def main():
    st.title("보행 시 장애물 안내 서비스")
    user_session = get_session_id()
    st.write(f"Session ID : {user_session}")
    uploaded_file = st.file_uploader("동영상을 선택하세요", type=["mp4"])

    if uploaded_file:
        fn = uploaded_file.name
        save_path = os.path.join(upload_path, fn)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            st.success(f"파일이 서버에 저장되었습니다.  {save_path} ")
        placeholder = st.empty()
        placeholder.info("GPU 서버로 파일 전송 중")
        # split_list = sorted(split_cut(save_path, 2))
        split_list = sorted(split_segment(save_path, 2))
        for idx, file in enumerate(split_list):
            placeholder.success(f"전송 완료, 파일 처리 중 {idx/len(split_list)*100:.1f} %")
            files = [("files", open(file, 'rb'))]
            req = requests.post(SERVER_URL, files=files)
            if req.status_code == 200:
                with open(os.path.join(tmp_rcv_path, f"{idx:04}.mp4"), 'wb') as f:
                    f.write(req.content)
            idx += 1
        placeholder.empty()
        rslt_file = os.path.join(dst_path, "result.mp4")
        concatenate(tmp_rcv_path, rslt_file)
        st.video(open(rslt_file, 'rb').read(), format="video/mp4")


dir_func(tmp_path)
dir_func(upload_path)
dir_func(tmp_rcv_path)
dir_func(dst_path)
main()
