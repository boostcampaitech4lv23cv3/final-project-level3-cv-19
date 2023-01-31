import os
from pathlib import Path
import shutil
from app.ffmpeg_func import split_cut, split_segment, concatenate
import app.substitute as subs
import streamlit.components.v1 as components
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


def dir_func(path: str, rmtree: bool = True, mkdir: bool = True):
    if rmtree:
        shutil.rmtree(path, ignore_errors=True)
    if mkdir:
        if not os.path.exists(path):
            os.makedirs(path)


def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")

    return ctx.session_id


def main():
    st.title("보행 시 장애물 안내 서비스")
    st.write(f"Session ID : {user_session}")
    if 'url' not in st.session_state:
        st.session_state.url = SERVER_URL
    URL_DISPLAY = st.empty()
    URL_DISPLAY.write(f'Current GPU Server URL is {st.session_state.url}')
    new_url = st.text_input("Put Servicing URL with 'http://' and Press ENTER", )
    if new_url != st.session_state.url:
        st.session_state.url = new_url
        URL_DISPLAY.write(f'Current GPU Server URL is {st.session_state.url}')
    uploaded_file = st.file_uploader("동영상을 선택하세요", type=["mp4"])

    if uploaded_file:
        fn = uploaded_file.name
        dir_func(upload_path, rmtree=False, mkdir=True)
        save_filepath = os.path.join(upload_path, fn)
        placeholder = st.empty()
        with open(save_filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            placeholder.success(f"파일이 서버에 저장되었습니다.")
        # split_list = sorted(split_cut(upload_path, 2))
        dir_func(tmp_path, rmtree=False, mkdir=True)
        dir_func(tmp_rcv_path, rmtree=False, mkdir=True)
        split_list = sorted(split_segment(save_filepath, 2))
        stat_code_checker = [False] * len(split_list)
        for idx, file in enumerate(split_list):
            placeholder.success(f"GPU 서버로 파일 전송, 파일 처리 중 {(idx+1) / len(split_list) * 100:.1f} % at {st.session_state.url}")
            files = [("files", open(os.path.join(tmp_path, file), 'rb'))]
            headers = {"filename": file, "user_id": user_session}
            response = requests.post(st.session_state.url, files=files, headers=headers)
            if response.status_code == 200:
                stat_code_checker[idx] = True
                head = response.headers
                fp = os.path.join(str(Path(tmp_rcv_path).parent), head.get('user_id'), head.get('filename'))
                print(fp)
                with open(fp, 'wb') as f:
                    f.write(response.content)
        placeholder.empty()
        if all(stat_code_checker):
            dir_func(dst_path, rmtree=False, mkdir=True)
            rslt_file = os.path.join(dst_path, "result.mp4")
            concatenate(tmp_rcv_path, rslt_file)
            st.video(open(rslt_file, 'rb').read(), format="video/mp4")
            subtitle_html = components.html()
            iframe = components.iframe()
            dir_func(tmp_path, rmtree=True, mkdir=False)
            dir_func(upload_path, rmtree=True, mkdir=False)
            dir_func(tmp_rcv_path, rmtree=True, mkdir=False)
        else:
            placeholder.warning("파일 송수신에 실패했습니다. 터미널 로그를 참조하세요")


user_session = get_session_id()
tmp_path = f"app/tmp/{user_session}/"
upload_path = f"app/uploaded/{user_session}/"
tmp_rcv_path = f"app/tmp_rcv/{user_session}/"
dst_path = f"app/result/{user_session}/"

main()
