import os
import pathlib
import json
from app.utils import dir_func

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)
    return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}.{int(millisecond):03d}"


def json2sub(session_id: str, json_str: str, fps=10, save: bool = True, ext: str = "vtt"):
    break_ = "\r\n" if ext == "srt" else "\n"
    vtt_header = "WEBVTT\n"
    frame_time_in_sec = 1/fps
    ts_start = .0
    ts_end = ts_start + frame_time_in_sec
    subtitle_arr = [] if ext == "srt" else [vtt_header]
    js_dict = json.loads(json_str)
    for frame_no, frame_items in sorted(js_dict.items(), key=lambda x: x[0], reverse=False):
        if len(frame_items):
            seq_line = f"{int(frame_no)}"
            time_line = f"{second_to_timecode(ts_start)} --> {second_to_timecode(ts_end)}"
            caption_line = break_.join([f"IDX: {obj_id} // WARNING: {obj_data['warning_lv']} // LOCATION: {obj_data['location']} // CLASS: {obj_data['class']}" for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False)])
            break_line = ""
            subtitle_arr.append("\n".join([seq_line, time_line, caption_line, break_line]))

            ts_start = ts_end
            ts_end += frame_time_in_sec

    parsed_str = "\n".join(subtitle_arr)
    if save:
        save_path = os.path.join(APP_PATH, "result", session_id)
        dir_func(save_path, rmtree=False, mkdir=True)
        with open(os.path.join(save_path, f"result.{ext}"), 'w+', encoding='utf-8') as f:
            f.writelines(parsed_str)
    return parsed_str


def get_html(external_ip: str, session_id: str, sub_ext: str = "vtt") -> tuple:
    html_path = os.path.join(APP_PATH, "templates", "html", session_id)
    dir_func(html_path, rmtree=True, mkdir=True)
    html_str = f"""
{{% extends "base.html" %}}
{{% block content %}}
  <div class="container">
    <video id="my-video" controls preload="auto" width="700" autoplay crossorigin="anonymous">
    <source src="http://{external_ip}:30002/{session_id}/video" type="video/mp4"/>
    <track src="http://{external_ip}:30002/{session_id}/subtitle" kind="subtitles" srclang="ko" label="한국어" default/>
  </video>
  </div>
{{% endblock %}}
"""
    html_fp = os.path.join(html_path, "subs.html")
    with open(html_fp, 'w+', encoding='utf-8') as f:
        f.writelines(html_str)
    return html_fp, html_str
