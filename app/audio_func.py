import os
import pathlib
import json
from utils import dir_func
from pydub import AudioSegment

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")


def json2audio(json_str: str, dst_path: str, fps=10, save: bool = True, ext: str = "wav"):
    class_names_index = {'bicycle': 0, 'bus': 1, 'car': 2, 'carrier': 3, 'cat': 4,
                         'dog': 5, 'motorcycle': 6, 'movable_signage': 7, 'person': 8, 'scooter': 9,
                         'stroller': 10, 'truck': 11, 'wheelchair': 12, 'barricade': 13, 'bench': 14,
                         'bollard': 15, 'chair': 16, 'fire_hydrant': 17, 'kiosk': 18, 'parking_meter': 19,
                         'pole': 20, 'potted_plant': 21, 'power_controller': 22, 'stop': 23, 'table': 24,
                         'traffic_light': 25, 'traffic_light_controller': 26, 'traffic_sign': 27, 'tree_trunk': 28}

    frame_time_in_sec = 1 / fps
    frame_time_in_ms = 1000 / fps
    ts_start = .0
    ts_end = ts_start + frame_time_in_sec
    ts_audio = .0
    js_dict = json.loads(json_str)
    silent_1sec = AudioSegment.from_wav(os.path.join(APP_PATH, "wav", "silent_1-second.wav"))
    silent_1frame = silent_1sec[:int(frame_time_in_ms)]  # (slicing is done by milliseconds)
    synthesized_audio = silent_1sec[0]  # 1ms
    for frame_no, frame_items in sorted(js_dict.items(), key=lambda x: x[0], reverse=False):
        found_obj = False
        if ts_start >= ts_audio:
            if len(frame_items):
                for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False):
                    if obj_data['warning_lv'] == "1":
                        # warning->location->class 순으로 인덱스 부여
                        # 클래스 29개 x 위치 3개 x 경고수준 2개 = 174개 wav
                        wav_index = 2 * int(obj_data['location']) + 6 * class_names_index[obj_data['class']]
                        if (wav_index <= 173) and (wav_index >= 0):
                            obj_warn_audio = AudioSegment.from_wav("./app/wav/" + str(wav_index) + ".wav")
                            gap_frame_in_ms = int(frame_time_in_ms) - (len(obj_warn_audio) % int(frame_time_in_ms))
                            obj_warn_audio_1frame = obj_warn_audio + silent_1sec[:gap_frame_in_ms - 1]
                            synthesized_audio += obj_warn_audio_1frame
                            ts_audio += len(obj_warn_audio_1frame) / 1000.0  # ms -> sec
                            found_obj = True
                            break

            if (found_obj is False) and (ts_start <= ts_audio):
                synthesized_audio += silent_1frame
                ts_audio += len(silent_1frame) / 1000.0  # ms -> sec

        ts_start = ts_end
        ts_end += frame_time_in_sec

    save_path = os.path.join(dst_path, f"synthesized_audio.{ext}")
    if save:
        dir_func(dst_path, rmtree=False, mkdir=True)
        synthesized_audio.export(save_path, format=ext)
