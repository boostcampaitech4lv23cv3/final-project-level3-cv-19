import os
import pathlib
import json
from collections import deque
from app.utils import dir_func

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")

# Detection Result Hashes
cls_hash = {'bicycle': '자전거', 'bus': '버스', 'car': '자동차', 'carrier': '손수레', 'cat': '고양이', 'dog': '개',
            'motorcycle': '오토바이', 'movable_signage': '간판', 'person': '사람', 'scooter': '스쿠터',
            'stroller': '유모차', 'truck': '트럭', 'wheelchair': '휠체어', 'barricade': '바리케이드', 'bench': '벤치',
            'bollard': '볼라드', 'chair': '의자', 'fire_hydrant': '소화전', 'kiosk': '키오스크',
            'parking_meter': '주차요금정산기', 'pole': '기둥', 'potted_plant': '화분', 'power_controller': '전력제어함',
            'stop': '정류장', 'table': '탁자', 'traffic_light': '신호등', 'traffic_light_controller': '신호등제어기',
            'traffic_sign': '교통표지판', 'tree_trunk': '가로수'}
postposition_hash = {'bicycle': '가', 'bus': '가', 'car': '가', 'carrier': '가', 'cat': '가', 'dog': '가',
                     'motorcycle': '가', 'movable_signage': '이', 'person': '이', 'scooter': '가', 'stroller': '가',
                     'truck': '이', 'wheelchair': '가', 'barricade': '가', 'bench': '가', 'bollard': '가', 'chair': '가',
                     'fire_hydrant': '이', 'kiosk': '가', 'parking_meter': '가', 'pole': '이', 'potted_plant': '이',
                     'power_controller': '이', 'stop': '이', 'table': '가', 'traffic_light': '이',
                     'traffic_light_controller': '가', 'traffic_sign': '이', 'tree_trunk': '가'}
loc_hash = {0: "왼쪽", 1: "중앙", 2: "오른쪽"}
warning_hash = {1: "가까이", 2: ""}


def frame_dict2caption_line(obj_data, warning_threshold=1):
    cls = obj_data["class"]
    location = int(obj_data["location"])
    warning = int(obj_data["warning_lv"])
    distance = float(obj_data["distance"])
    heading = float(obj_data["heading"])
    if warning <= warning_threshold:
        return f"{cls_hash[cls]}{postposition_hash[cls]} {loc_hash[location]}에 {warning_hash[warning]} 있습니다."
    else:
        return ""


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
    subtitle_arr = [] if ext == "srt" else [vtt_header]
    js_dict = json.loads(json_str)
    caption_queue = deque()
    for frame_no, frame_items in sorted(js_dict.items(), key=lambda x: x[0], reverse=False):
        if len(frame_items):
            caption_line_arr = []
            for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False):
                tmp_caption_line = frame_dict2caption_line(obj_data=obj_data)
                if len(tmp_caption_line.strip()):
                    caption_line_arr.append(tmp_caption_line)
            if len(caption_line_arr):
                caption_line = break_.join(caption_line_arr)
                caption_queue.append((int(frame_no), caption_line))

    while len(caption_queue) > 1:
        caption_idx = 1
        first_frame, caption_line = caption_queue.popleft()
        second_frame, _ = caption_queue[0]
        ts_start = (first_frame - 1) * frame_time_in_sec
        duration = (second_frame - first_frame) * frame_time_in_sec
        ts_end = (ts_start + duration) if duration <= 1 else (ts_start + 1)
        time_line = f"{second_to_timecode(ts_start)} --> {second_to_timecode(ts_end)}"
        subtitle_arr.append("\n".join([str(caption_idx), time_line, caption_line, ""]))
        caption_idx += 1
        if len(caption_queue) == 1:
            last_frame, caption_line = caption_queue.popleft()
            ts_start = (last_frame - 1) * frame_time_in_sec
            ts_end = ts_start + 1
            time_line = f"{second_to_timecode(ts_start)} --> {second_to_timecode(ts_end)}"
            subtitle_arr.append("\n".join([str(caption_idx), time_line, caption_line, ""]))

    parsed_str = "\n".join(subtitle_arr)
    if save:
        save_path = os.path.join(APP_PATH, "result", session_id)
        dir_func(save_path, rmtree=False, mkdir=True)
        with open(os.path.join(save_path, f"result.{ext}"), 'w+', encoding='utf-8') as f:
            f.writelines(parsed_str)
    return parsed_str
