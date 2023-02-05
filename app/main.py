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

from requests import get

import shlex
from subprocess import check_call, PIPE
import json
from pathlib import Path

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
tmp_path = f"app/tmp/{user_session}/"
wav_path = f"app/audio/{user_session}/"
tensorrtmodel_file = f"yolov8n_custom_int8.trt"
#tensorrtmodel_file = f"yolov8n_custom_fp16.trt"
#tensorrtmodel_file = f"yolov8n_custom_fp32.trt"

PRJ_ROOT_PATH = Path(__file__).parent.parent.absolute()
MODEL_DIR = os.path.join(PRJ_ROOT_PATH, "Model")
TENSORRT_DIR = os.path.join(MODEL_DIR, "onnx_tensorrt")

def main():
    st.title("보행 시 장애물 안내 서비스")
    st.write(f"Session ID : {user_session}")

    placeholder = st.empty()
    upload_place = st.empty()
    uploaded_file = upload_place.file_uploader("동영상을 선택하세요", type=["mp4"])

    if uploaded_file:
        # Save Uploaded File
        dir_func(upload_path, rmtree=True, mkdir=True)
        fn = uploaded_file.name.replace(" ", "_")
        save_filepath = os.path.join(upload_path, fn)
        with open(save_filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            placeholder.success(f"파일이 서버에 저장되었습니다.")
            upload_place.empty()
        dir_func(tmp_path, rmtree=True, mkdir=True)
        preprocessed_file = os.path.join(tmp_path, "preprocessed.mp4")
        result_av_file = os.path.join(dst_path, "result.mp4")

        col_slide, col_button = st.columns([2, 1])
        slide_value = col_slide.slider("Confidence Lv Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
        button_value = col_button.button("Start Process")

        if button_value:
            col_slide.empty()

            try:
                with st.spinner("동영상 전처리 중..."):
                    video_preprocessing(save_filepath, preprocessed_file, resize_h=640, tgt_framerate=TARGET_FPS)
                placeholder.success("동영상 전처리 완료")
                with st.spinner("객체 탐지 중..."):
                    if 1:  # Pytorch
                        from Model.detector import detect
                        img_dir = os.path.join(tmp_path, "img_dir")
                        dir_func(img_dir, rmtree=True, mkdir=True)
                        frame_json = detect(preprocessed_file, user_session, conf_thres=slide_value)
                    else:  # TensorRT
                        img_dir = os.path.join(tmp_path, "img_dir")
                        dir_func(img_dir, rmtree=True, mkdir=True)
                        tensorrt_file_path = os.path.join(TENSORRT_DIR, 'trtinference.py')
                        preprocessing_file_path = os.path.join(tmp_path, 'preprocessed.mp4')
                        tensorrt_model_file_path = os.path.join(TENSORRT_DIR, tensorrtmodel_file)
                        cmd = f"python {tensorrt_file_path} -v {preprocessing_file_path} -e {tensorrt_model_file_path} -s {user_session} -c {slide_value}"
                        check_call(shlex.split(cmd), universal_newlines=True)
                        JSON_FILE = os.path.join(tmp_path, 'objdetection.json')
                        with open(JSON_FILE, "rb") as json_file:
                            frame_json_dict = json.load(json_file)
                            frame_json = json.dumps(frame_json_dict)
                        
                placeholder.success("객체 탐지 완료")
                json2sub(session_id=user_session, json_str=frame_json, fps=TARGET_FPS, save=True)
                json2audio(dst_path=wav_path, json_str=frame_json, fps=TARGET_FPS, save=True)
                audio_file = os.path.join(wav_path, "synthesized_audio.wav")
                dir_func(dst_path, rmtree=False, mkdir=True)
                with st.spinner("객체 탐지 결과 종합 중..."):
                    combine_video_audio(img_dir, audio_file, result_av_file, fps=TARGET_FPS)
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
                dir_func(wav_path, rmtree=True, mkdir=False)


dir_func(dst_path, rmtree=True, mkdir=True)
main()
