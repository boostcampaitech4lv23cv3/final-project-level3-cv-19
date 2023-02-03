import os
import sys
import logging
import streamlit as st
import streamlit.components.v1 as components

from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from app.utils import dir_func
from app.ffmpeg_func import video_preprocessing, combine_video_audio
from app.subtitle_func import json2sub
from app.audio_func import json2audio
from Model.detector import detect

from requests import get

# if st.session_state.get("a", False):
#     st.session_state.disabled = True
# elif st.session_state.get("a", True):
#     st.session_state.disabled = False

# def disable(b, c):
#     st.session_state["disabled"] = b
#     st.session_state["disabled"] = c


def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")
    return ctx.session_id


TARGET_FPS = 15
EXTERNAL_IP = get('https://api.ipify.org').content.decode('utf8')
user_session = get_session_id()
st.set_page_config(layout="centered")
container_w = 700
subtitle_ext = "vtt"

# PATH SETTINGS
upload_path = f"app/uploaded/{user_session}/"
dst_path = f"app/result/{user_session}/"
html_path = f"app/html/{user_session}/"
tmp_path = f"app/tmp/{user_session}/"
wav_path = f"app/audio/{user_session}/"


def main():
    st.title("보행 시 장애물 안내 서비스")
    st.write(f"Session ID : {user_session}")

    # uploaded_file = st.file_uploader("동영상을 선택하세요", type=["mp4"], key="c", disabled=st.session_state.get("disabled", True))
    placeholder = st.empty()
    upload_place = st.empty()
    uploaded_file = upload_place.file_uploader("동영상을 선택하세요", type=["mp4"])

    if uploaded_file:
        # Save Uploaded File
        dir_func(upload_path, rmtree=True, mkdir=True)
        fn = uploaded_file.name
        save_filepath = os.path.join(upload_path, fn)
        with open(save_filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            placeholder.success(f"파일이 서버에 저장되었습니다.")
            upload_place.empty()
        dir_func(tmp_path, rmtree=True, mkdir=True)
        preprocessed_file = os.path.join(tmp_path, "preprocessed.mp4")
        result_video_file = os.path.join(dst_path, "resultvideo.mp4")
        result_av_file = os.path.join(dst_path, "result.mp4")

        col_slide, col_button = st.columns([2, 1])
        # slide_value = col_slide.slider("Confidence Lv Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05, key="b", disabled=st.session_state.get("disabled", True))
        # button_value = col_button.button("Start Process", key="a", on_click=disable, args=(True, True))
        slide_value = col_slide.slider("Confidence Lv Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
        button_value = col_button.button("Start Process")

        if button_value:
            try:
                video_preprocessing(save_filepath, preprocessed_file, resize_h=640, tgt_framerate=TARGET_FPS)
                placeholder.success("동영상 전처리 완료")
                with st.spinner("객체 탐지 중..."):
                    if 1:  # Pytorch
                        result_video_file, frame_json = detect(preprocessed_file, user_session, result_video_file, conf_thres=slide_value)
                    else:  # TensorRT
                        from Model.onnx_tensorrt.utils import BaseEngine
                        pred = BaseEngine(engine_path='./Model/onnx_tensorrt/yolov8n_custom.trt')
                        result_video_file, frame_json = pred.detect_video(file_name=preprocessed_file, user_session=user_session, conf=0.1, end2end=True)
                placeholder.success("객체 탐지 완료, 후처리 중...")
                json2sub(session_id=user_session, json_str=frame_json, fps=TARGET_FPS, save=True)
                json2audio(dst_path=wav_path, json_str=frame_json, fps=TARGET_FPS, save=True)
                audio_file = os.path.join(wav_path, "synthesized_audio.wav")
                combine_video_audio(result_video_file, audio_file, result_av_file)
                components.html(f"""
                  <div class="container">
                    <video controls preload="auto" width="{container_w}" autoplay crossorigin="anonymous">
                      <source src="http://{EXTERNAL_IP}:30002/{user_session}/video" type="video/mp4"/>
                      <track src="http://{EXTERNAL_IP}:30002/{user_session}/subtitle" srclang="ko" type="text/{subtitle_ext}" default/>
                  </video>
                  </div>
                """, width=container_w, height=int(container_w / 16 * 9))
                placeholder.success("처리 완료")

            except Exception as e:
                placeholder.warning(f"파일 처리 중 오류가 발생하였습니다.\n{e.with_traceback(sys.exc_info()[2])}")
                logging.exception(str(e), exc_info=True)

            finally:
                dir_func(upload_path, rmtree=True, mkdir=False)
                dir_func(tmp_path, rmtree=True, mkdir=False)


dir_func(dst_path, rmtree=True, mkdir=True)
main()
