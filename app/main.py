"""
Commented Codes are No More Used from 01Feb2023 because of Server Integration.
"""
import os
import sys
import logging
import streamlit as st
import streamlit.components.v1 as components

from app.utils import dir_func
from app.ffmpeg_func import video_preprocessing
from app.subtitle_func import get_html, json2srt
from Model.detector import detect


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


def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")
    return ctx.session_id


SERVER_URL = "http://localhost:30002/results"
TARGET_FPS = 15
user_session = get_session_id()
st.set_page_config(layout="centered")

# PATH SETTINGS
upload_path = f"app/uploaded/{user_session}/"
dst_path = f"app/result/{user_session}/"
html_path = f"app/html/{user_session}/"
tmp_path = f"app/tmp/{user_session}/"


def main():
    st.title("보행 시 장애물 안내 서비스")
    st.write(f"Session ID : {user_session}")

    uploaded_file = st.file_uploader("동영상을 선택하세요", type=["mp4"])
    placeholder = st.empty()

    if uploaded_file:
        # Save Uploaded File
        dir_func(upload_path, rmtree=True, mkdir=True)
        fn = uploaded_file.name
        save_filepath = os.path.join(upload_path, fn)
        with open(save_filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            placeholder.success(f"파일이 서버에 저장되었습니다.")

        dir_func(tmp_path, rmtree=True, mkdir=True)
        dir_func(dst_path, rmtree=True, mkdir=True)
        preprocessed_file = os.path.join(tmp_path, "preprocessed.mp4")
        result_file = os.path.join(dst_path, "result.mp4")

        try:
            video_preprocessing(save_filepath, preprocessed_file, resize_h=640, tgt_framerate=TARGET_FPS)
            result_file, frame_json = detect(preprocessed_file, user_session, result_file)
            json2srt(session_id=user_session, json_str=frame_json, fps=TARGET_FPS, save=True)
            st.video(open(result_file, 'rb').read(), format="video/mp4")
            # TODO
            # components.html(get_html(session_id=user_session)[1], width=700)

        except Exception as e:
            placeholder.warning(f"파일 처리 중 요류가 발생하였습니다.\n{e.with_traceback(sys.exc_info()[2])}")
            logging.exception(str(e), exc_info=True)

        finally:
            dir_func(upload_path, rmtree=False, mkdir=False)
            dir_func(tmp_path, rmtree=False, mkdir=False)


main()
